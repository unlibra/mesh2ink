# SPDX-License-Identifier: GPL-3.0-or-later
"""テスト用のシーンを作って .blend に保存する。形もテクスチャもこのスクリプトが作る。

使い方：blender --background --factory-startup --python make_test_scene.py -- <出力.blend> [<種類>]

種類を付けると、条件に合わないシーンを作る。止まり方のテストに使う。
no_camera（カメラが無い）、no_mesh（メッシュが無い）、no_material（材質の無い面がある）、
too_many_parts（材質が65個ある）、away（カメラが物の無い方を向いている）、plain（単色の球だけで、顔が無い）

入るもの：顔の描き込み（まつ毛、眉、口、頬の赤み）をテクスチャに持つ頭、
瞳のテクスチャを持つ目、単色の髪、材質が2つの体、カメラ。
"""
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Vector

HEAD_R = 0.1
HEAD_Z = 1.5
EYE_U = 0.19  # 顔のテクスチャ上での、目の中心の左右の位置（中央からの距離）
EYE_R = 0.019


def image_from_array(name, rgb):
    """rgb は上が先頭の (高さ, 幅, 3) の配列。値は0〜1。"""
    h, w = rgb.shape[:2]
    img = bpy.data.images.new(name, w, h, alpha=False)
    rgba = np.concatenate([rgb[::-1], np.ones((h, w, 1))], axis=2).astype(np.float32)
    img.pixels.foreach_set(rgba.ravel())
    img.pack()
    return img


def face_texture(n=512):
    v, u = np.mgrid[0:n, 0:n] / (n - 1)
    v = 1 - v  # 上が1
    rgb = np.empty((n, n, 3))
    rgb[:] = (1.0, 0.87, 0.80)
    for side in (-1, 1):
        cx = 0.5 + side * EYE_U
        # 頬の赤み。広くてぼんやりしている
        blush = np.exp(-(((u - (0.5 + side * 0.25)) ** 2 + (v - 0.38) ** 2) / (2 * 0.05**2)))
        rgb += blush[..., None] * (np.array((1.0, 0.62, 0.60)) - rgb) * 0.7
        # まつ毛。目の上側の太い弧
        r = np.hypot(u - cx, v - 0.5)
        lash = (np.abs(r - 0.105) < 0.013) & (v - 0.5 > 0.03)
        rgb[lash] = (0.12, 0.08, 0.10)
        # 眉。細い弧
        brow = (np.abs(u - cx) < 0.09) & (np.abs(v - (0.70 - 2.0 * (u - cx) ** 2)) < 0.006)
        rgb[brow] = (0.40, 0.24, 0.18)
    # 口。細い弧
    mouth = (np.abs(u - 0.5) < 0.08) & (np.abs(v - (0.24 + 4.0 * (u - 0.5) ** 2)) < 0.005)
    rgb[mouth] = (0.62, 0.25, 0.25)
    return rgb


def eye_texture(n=128):
    v, u = np.mgrid[0:n, 0:n] / (n - 1)
    v = 1 - v
    r = np.hypot(u - 0.5, v - 0.5) / 0.5
    rgb = np.empty((n, n, 3))
    rgb[:] = (0.97, 0.97, 0.97)
    iris = r < 0.8
    t = np.clip(r / 0.8, 0, 1)[..., None]
    rgb[iris] = (np.array((0.10, 0.25, 0.60)) * (1 - t) + np.array((0.25, 0.55, 0.90)) * t)[iris]
    rgb[r < 0.28] = (0.06, 0.06, 0.14)
    rgb[np.hypot(u - 0.36, v - 0.66) < 0.07] = (1.0, 1.0, 1.0)
    return rgb


def material(name, color=None, image=None):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    if image is not None:
        tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
        tex.image = image
        mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    return mat


def add_object(name, bm, materials, location):
    for f in bm.faces:
        f.smooth = True
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for m in materials:
        me.materials.append(m)
    ob = bpy.data.objects.new(name, me)
    ob.location = location
    bpy.context.scene.collection.objects.link(ob)
    return ob


def project_front_uv(bm, radius_x, radius_z):
    """-Yの側から見た正面の投影でUVを付ける。"""
    uv = bm.loops.layers.uv.new("UV")
    for f in bm.faces:
        for loop in f.loops:
            co = loop.vert.co
            loop[uv].uv = (co.x / radius_x * 0.5 + 0.5, co.z / radius_z * 0.5 + 0.5)


def add_camera(scene, location=(0.3, -1.6, 1.5), target=(0, 0, 1.43)):
    cam_data = bpy.data.cameras.new("camera")
    cam_data.lens = 120
    cam = bpy.data.objects.new("camera", cam_data)
    cam.location = location
    cam.rotation_euler = (Vector(target) - cam.location).to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(cam)
    scene.camera = cam
    scene.render.resolution_x = scene.render.resolution_y = 1024


def add_ball(name, materials, location=(0, 0, HEAD_Z)):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=8, radius=HEAD_R)
    return add_object(name, bm, materials, location)


def main(out_path, variant):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    if variant is not None:
        if variant != "no_camera":
            add_camera(scene, target=(0, -3, 1.5) if variant == "away" else (0, 0, 1.43))
        if variant == "no_material":
            add_ball("bare", [])
        elif variant == "too_many_parts":
            ball = add_ball("many", [material(f"m{i:02d}", color=(0.5, 0.5, 0.5)) for i in range(65)])
            for i, polygon in enumerate(ball.data.polygons):
                polygon.material_index = i % 65
        elif variant != "no_mesh":
            add_ball("ball", [material("plain", color=(0.8, 0.6, 0.4))])
        bpy.ops.wm.save_as_mainfile(filepath=out_path)
        return

    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, radius=HEAD_R)
    project_front_uv(bm, HEAD_R, HEAD_R)
    face_mat = material("face", image=image_from_array("face_tex", face_texture()))
    add_object("head", bm, [face_mat], (0, 0, HEAD_Z))

    eye_mat = material("eye", image=image_from_array("eye_tex", eye_texture()))
    for side in (-1, 1):
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=12, radius=EYE_R)
        project_front_uv(bm, EYE_R, EYE_R)
        bmesh.ops.scale(bm, vec=(1, 0.3, 1), verts=bm.verts)
        x = side * EYE_U * 2 * HEAD_R
        y = -((HEAD_R**2 - x**2) ** 0.5)
        add_object(f"eye_{'L' if side > 0 else 'R'}", bm, [eye_mat], (x, y, HEAD_Z))

    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, radius=HEAD_R * 1.08)
    front_low = [f for f in bm.faces if f.calc_center_median().y < 0.0 and f.calc_center_median().z < 0.045]
    bmesh.ops.delete(bm, geom=front_low, context="FACES")
    add_object("hair", bm, [material("hair", color=(0.45, 0.28, 0.20))], (0, 0, HEAD_Z))

    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=32, radius1=0.13, radius2=0.035, depth=0.26)
    bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], plane_co=(0, 0, 0.04), plane_no=(0, 0, 1))
    for f in bm.faces:
        f.material_index = 0 if f.calc_center_median().z > 0.04 else 1
    # 2つ目の材質は、英数字でない名前の扱いを試すために日本語の名前にしている
    add_object("body", bm, [material("body", color=(1.0, 0.87, 0.80)), material("服", color=(0.30, 0.45, 0.75))], (0, 0, 1.28))

    add_camera(scene)
    bpy.ops.wm.save_as_mainfile(filepath=out_path)


_args = sys.argv[sys.argv.index("--") + 1 :]
main(_args[0], _args[1] if len(_args) > 1 else None)
