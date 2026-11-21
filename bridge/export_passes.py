# SPDX-License-Identifier: GPL-3.0-or-later
"""Blenderの中で動く橋渡し。開いているシーンから、画像の組（passes）を書き出す。

使い方：blender --background <シーン.blend> --python export_passes.py -- <設定.json> <出力のフォルダ>

設定は {"size", "subdivision", "line_width", "line_smooth"} を持つJSON。
シーンは保存しない。変更はこのプロセスのメモリの中だけで行う。
シーンが条件に合わない時は、出力のフォルダに error.txt だけを書いて終わる。
"""
import json
import math
import struct
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

FORMAT_VERSION = 1
MAX_PARTS = 64
PART_LEVELS = (0, 64, 128, 191, 255)
# カメラから見た光の来る向き（右、上、手前）。影のレイヤー用の light.png に使う
LIGHT_DIRECTION = (-0.3, 0.5, 0.85)
KEEP_CHUNKS = {b"IHDR", b"PLTE", b"tRNS", b"IDAT", b"IEND"}


class SceneError(Exception):
    pass


def part_color(part_id):
    """部位の番号（1から）を、parts.png での色（8ビットのRGB）にする。"""
    return [PART_LEVELS[part_id % 5], PART_LEVELS[part_id // 5 % 5], PART_LEVELS[part_id // 25 % 5]]


def strip_png_metadata(path):
    """PNGから、絵に要るチャンク以外（パスや日時の入るテキスト）を消す。"""
    raw = path.read_bytes()
    out = [raw[:8]]
    pos = 8
    while pos < len(raw):
        (length,) = struct.unpack(">I", raw[pos : pos + 4])
        kind = raw[pos + 4 : pos + 8]
        end = pos + 12 + length
        if kind in KEEP_CHUNKS:
            out.append(raw[pos:end])
        pos = end
    path.write_bytes(b"".join(out))


def collect_slots(meshes):
    """(物, スロットの番号, 材質) の一覧。面が使っているスロットだけ。材質の無い面があれば止まる。"""
    slots = []
    for ob in meshes:
        polygons = ob.data.polygons
        if not polygons:
            continue
        used = sorted({min(p.material_index, max(len(ob.material_slots) - 1, 0)) for p in polygons})
        for i in used:
            if i >= len(ob.material_slots) or ob.material_slots[i].material is None:
                raise SceneError(f"材質の無い面を持つメッシュがあります：{ob.name}")
            slots.append((ob, i, ob.material_slots[i].material))
    return slots


def emission_material(name, setup=None, color=(0.0, 0.0, 0.0)):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*color, 1.0)
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    if setup:
        setup(nt, em)
    return mat


def color_input(node):
    """シェーダーのノードから、地の色の入力を探す。混ぜるノードは、中をたどる。"""
    if node.type in {"MIX_SHADER", "ADD_SHADER"}:
        for sock in node.inputs:
            if sock.type == "SHADER" and sock.is_linked:
                found = color_input(sock.links[0].from_node)
                if found is not None:
                    return found
        return None
    if node.type == "BSDF_TRANSPARENT":
        return None
    for name in ("Base Color", "Color"):
        if name in node.inputs:
            return node.inputs[name]
    return None


def color_material(src):
    """材質の写しを作り、地の色（テクスチャか単色）だけを陰影なしで出すようにつなぎ直す。"""
    mat = src.copy()
    nt = mat.node_tree
    sock = None
    if nt is not None:
        out = nt.get_output_node("EEVEE")
        if out is not None and out.inputs["Surface"].is_linked:
            sock = color_input(out.inputs["Surface"].links[0].from_node)
    if sock is None:
        return emission_material(f"m2i_color_{src.name}", color=tuple(src.diffuse_color[:3]))
    em = nt.nodes.new("ShaderNodeEmission")
    if sock.is_linked:
        nt.links.new(sock.links[0].from_socket, em.inputs["Color"])
    else:
        em.inputs["Color"].default_value = sock.default_value
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    mat.surface_render_method = "DITHERED"
    return mat


def normal_material():
    def setup(nt, em):
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        tr = nt.nodes.new("ShaderNodeVectorTransform")
        tr.vector_type, tr.convert_from, tr.convert_to = "NORMAL", "WORLD", "CAMERA"
        mad = nt.nodes.new("ShaderNodeVectorMath")
        mad.operation = "MULTIPLY_ADD"
        mad.inputs[1].default_value = (0.5, 0.5, 0.5)
        mad.inputs[2].default_value = (0.5, 0.5, 0.5)
        nt.links.new(geo.outputs["Normal"], tr.inputs[0])
        nt.links.new(tr.outputs[0], mad.inputs[0])
        nt.links.new(mad.outputs[0], em.inputs["Color"])

    return emission_material("m2i_normal", setup)


def depth_material(near, far):
    def setup(nt, em):
        cam = nt.nodes.new("ShaderNodeCameraData")
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = near, far
        nt.links.new(cam.outputs["View Z Depth"], mr.inputs["Value"])
        nt.links.new(mr.outputs[0], em.inputs["Color"])

    return emission_material("m2i_depth", setup)


def light_material():
    mat = bpy.data.materials.new("m2i_light")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    bsdf = nt.nodes.new("ShaderNodeBsdfDiffuse")
    bsdf.inputs["Color"].default_value = (1, 1, 1, 1)
    nt.links.new(bsdf.outputs[0], nt.nodes.new("ShaderNodeOutputMaterial").inputs["Surface"])
    return mat


def flat_world(name, value):
    world = bpy.data.worlds.new(name)
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (value, value, value, 1)
    nt.links.new(bg.outputs[0], nt.nodes.new("ShaderNodeOutputWorld").inputs["Surface"])
    return world


def depth_range(cam, meshes):
    """写る物の囲みの箱から、カメラからの距離の最小と最大を出す。"""
    inv = cam.matrix_world.inverted()
    depths = [-(inv @ (ob.matrix_world @ Vector(c))).z for ob in meshes for c in ob.bound_box]
    near, far = min(depths), max(depths)
    pad = (far - near) * 0.01 + 1e-6
    return near - pad, far + pad


def pixel_size(scene, cam, depth, width):
    """カメラから depth の距離にある面での、1画素の長さ（m）。"""
    m = cam.matrix_world
    a = world_to_camera_view(scene, cam, m @ Vector((0, 0, -depth)))
    b = world_to_camera_view(scene, cam, m @ Vector((1, 0, -depth)))
    return 1.0 / (abs(b.x - a.x) * width)


def setup_render(scene, size):
    r = scene.render
    long_side = max(r.resolution_x, r.resolution_y)
    r.resolution_x, r.resolution_y = (max(1, round(size * v / long_side)) for v in (r.resolution_x, r.resolution_y))
    r.resolution_percentage = 100
    r.pixel_aspect_x = r.pixel_aspect_y = 1.0
    r.engine = "BLENDER_EEVEE"
    r.use_border = False
    r.use_compositing = r.use_sequencer = False
    r.use_freestyle = False
    r.use_motion_blur = False
    r.use_stamp = False
    r.dither_intensity = 0.0
    r.use_file_extension = True
    r.image_settings.file_format = "PNG"
    r.image_settings.compression = 15
    scene.display_settings.display_device = "sRGB"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.view_settings.use_curve_mapping = False


def main(config_path, out_dir):
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    scene = bpy.context.scene
    cam = scene.camera
    if cam is None or cam.type != "CAMERA":
        raise SceneError("有効なカメラがありません。シーンにカメラを置き、有効なカメラに設定してください")
    view_objects = list(bpy.context.view_layer.objects)
    meshes = [o for o in view_objects if o.type == "MESH" and not o.hide_render and o.data.polygons]
    if not meshes:
        raise SceneError("カメラに写る物がありません。描画で表示されるメッシュがありません")
    slots = collect_slots(meshes)
    materials = sorted({m for _, _, m in slots}, key=lambda m: m.name)
    if len(materials) > MAX_PARTS:
        raise SceneError(f"部位が多すぎます：材質が{len(materials)}個あります。上限は{MAX_PARTS}個です")
    part_ids = {m: i + 1 for i, m in enumerate(materials)}

    setup_render(scene, cfg["size"])
    width, height = scene.render.resolution_x, scene.render.resolution_y
    # メッシュ以外の描画される物（文字、カーブ、ほかの線画、光）は、画像の組に入れない
    for ob in view_objects:
        if ob.type == "MESH":
            if ob.hide_render:
                ob.lineart.usage = "EXCLUDE"
        elif ob is not cam:
            ob.hide_render = True
    # 面を割ってなめらかにする。線とほかの画像の形をそろえるため、最初に入れる
    if cfg["subdivision"]:
        for ob in meshes:
            sd = ob.modifiers.new("m2i_subdivision", "SUBSURF")
            sd.levels = sd.render_levels = cfg["subdivision"]
    bpy.context.view_layer.update()
    near, far = depth_range(cam, meshes)
    px = pixel_size(scene, cam, (near + far) / 2, width)

    def render(name, pick, *, raw, aa, depth16=False, mode="RGBA", transparent=True, world=None):
        for ob, i, mat in slots:
            ob.material_slots[i].material = pick(mat)
        scene.world = world
        scene.render.film_transparent = transparent
        scene.view_settings.view_transform = "Raw" if raw else "Standard"
        scene.render.image_settings.color_mode = mode
        scene.render.image_settings.color_depth = "16" if depth16 else "8"
        scene.render.filter_size = 1.5 if aa else 0.01
        scene.eevee.taa_render_samples = 16 if aa else 1
        path = out_dir / f"{name}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        strip_png_metadata(path)

    black = flat_world("m2i_black", 0.0)
    color_mats = {m: color_material(m) for m in materials}
    render("color", color_mats.get, raw=False, aa=True, world=black)
    flat_mats = {m: emission_material(f"m2i_part_{i}", color=tuple(v / 255 for v in part_color(i))) for m, i in part_ids.items()}
    render("parts", flat_mats.get, raw=True, aa=False, world=black)
    depth_mat = depth_material(near, far)
    render("depth", lambda m: depth_mat, raw=True, aa=True, depth16=True, world=black)
    normal_mat = normal_material()
    render("normal", lambda m: normal_mat, raw=True, aa=True, depth16=True, world=black)

    sun_data = bpy.data.lights.new("m2i_sun", "SUN")
    sun_data.energy = 3.0
    sun_data.angle = math.radians(1)
    sun = bpy.data.objects.new("m2i_sun", sun_data)
    scene.collection.objects.link(sun)
    sun.rotation_euler = (cam.matrix_world.to_quaternion() @ Vector(LIGHT_DIRECTION).normalized().to_track_quat("Z", "Y")).to_euler()
    light_mat = light_material()
    render("light", lambda m: light_mat, raw=False, aa=True, world=black)
    sun.hide_render = True

    # 形の線：Grease PencilのLine Artモディファイア。白い面と白い背景の上に、黒い線だけを描く
    gp = bpy.data.grease_pencils.new("m2i_lineart")
    gp_ob = bpy.data.objects.new("m2i_lineart", gp)
    scene.collection.objects.link(gp_ob)
    layer = gp.layers.new("lines")
    layer.frames.new(scene.frame_current)
    layer.use_lights = False
    ink = bpy.data.materials.new("m2i_ink")
    bpy.data.materials.create_gpencil_data(ink)
    ink.grease_pencil.color = (0, 0, 0, 1)
    gp.materials.append(ink)
    la = gp_ob.modifiers.new("m2i_lineart", "LINEART")
    la.source_type = "SCENE"
    la.target_layer = "lines"
    la.target_material = ink
    la.use_crease = la.use_intersection = la.use_loose = la.use_edge_mark = False
    la.radius = cfg["line_width"] * px
    gp_ob.show_in_front = True
    if cfg["line_smooth"]:
        sm = gp_ob.modifiers.new("m2i_smooth", "GREASE_PENCIL_SMOOTH")
        sm.factor, sm.step, sm.use_keep_shape = 1.0, cfg["line_smooth"], True
    white = flat_world("m2i_white", 1.0)
    white_mat = emission_material("m2i_white", color=(1.0, 1.0, 1.0))
    for name, contour in (("line_contour", True), ("line_material", False)):
        la.use_contour, la.use_material = contour, not contour
        render(name, lambda m: white_mat, raw=False, aa=True, mode="BW", transparent=False, world=white)

    meta = {
        "version": FORMAT_VERSION,
        "width": width,
        "height": height,
        "projection": cam.data.type,
        "pixel_size": float(f"{px:.6g}"),
        "depth_near": float(f"{near:.6g}"),
        "depth_far": float(f"{far:.6g}"),
        "parts": [{"id": i, "color": part_color(i), "material": m.name} for m, i in part_ids.items()],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def entry():
    args = sys.argv[sys.argv.index("--") + 1 :]
    out_dir = Path(args[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        main(args[0], out_dir)
    except SceneError as e:
        (out_dir / "error.txt").write_text(str(e), encoding="utf-8")


entry()
