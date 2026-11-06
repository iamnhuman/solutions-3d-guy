"""Build the editable clay character and the rig used by the browser.

blender --background --factory-startup --python-exit-code 1 \
  --python scripts/build-character.py -- [--preview-dir /path/to/renders]
"""

import argparse
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


ROOT = Path(__file__).resolve().parents[1]
TAU = 2 * math.pi
ELEVATION = 0.185
PIXELS_PER_UNIT = 242.75
FRAME_HEIGHT = 1350 / PIXELS_PER_UNIT
FRAME_CENTER_X = 19 / PIXELS_PER_UNIT
FRAME_CENTER_Z = (1070 - 675) / PIXELS_PER_UNIT / math.cos(ELEVATION)
HEAD_CENTER = 2.239
HEAD_WIDTH = 0.503
HEAD_HEIGHT = 0.513
HEAD_DEPTH = 0.463
HEAD_PIVOT = (0.0, 0.006, 1.823)


def smoothstep(lo, hi, value):
    t = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    return t * t * (3 - 2 * t)


def gray_material(name, value, roughness, specular=0.25):
    material = bpy.data.materials.new(name)
    material.diffuse_color = (value, value, value, 1)
    material.use_nodes = True
    shader = material.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value = material.diffuse_color
    shader.inputs['Roughness'].default_value = roughness
    shader.inputs['Specular IOR Level'].default_value = specular
    return material


def mesh_object(name, vertices, faces, material=None):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    if material is not None:
        obj.data.materials.append(material)
    return obj


def section_rows(anchors, step):
    """Monotone cubic interpolation keeps soles and taper free of overshoot."""
    rows = np.asarray(anchors, dtype=float)
    levels, values = rows[:, 0], rows[:, 1:]
    spans = np.diff(levels)
    slopes = np.diff(values, axis=0) / spans[:, None]
    tangents = np.zeros_like(values)
    tangents[0], tangents[-1] = slopes[0], slopes[-1]
    for i, (a, b) in enumerate(zip(slopes[:-1], slopes[1:]), start=1):
        use = a * b > 0
        tangents[i, use] = 2 * a[use] * b[use] / (a[use] + b[use])
    for i, dz in enumerate(spans):
        count = max(1, math.ceil(dz / step))
        for j in range(count):
            t = j / count
            a, b = values[i], values[i + 1]
            p, q = dz * tangents[i], dz * tangents[i + 1]
            value = ((2*t**3 - 3*t*t + 1)*a + (t**3 - 2*t*t + t)*p
                     + (-2*t**3 + 3*t*t)*b + (t**3 - t*t)*q)
            yield (levels[i] + t * dz, *value)
    yield tuple(rows[-1])


def loft(name, rows, material, segments=48, step=0.028):
    vertices = []
    for z, cx, cy, rx, ry, power in section_rows(rows, step):
        for i in range(segments):
            angle = TAU * i / segments
            c, s = math.cos(angle), math.sin(angle)
            x = cx + rx * math.copysign(abs(c)**power, c)
            y = cy + ry * math.copysign(abs(s)**power, s)
            vertices.append((x, y, z))
    faces = [tuple(reversed(range(segments)))]
    for offset in range(0, len(vertices) - segments, segments):
        for i in range(segments):
            a, b = offset + i, offset + (i + 1) % segments
            faces.append((a, b, b + segments, a + segments))
    faces.append(tuple(range(len(vertices) - segments, len(vertices))))
    return mesh_object(name, vertices, faces, material)


def activate(objects, active):
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = active


def blend_forms(name, pieces, material, voxel_size, simplify, smoothing, hip_blend=False):
    obj = pieces[0]
    activate(pieces, obj)
    bpy.ops.object.join()
    obj.name = name
    obj.data.name = name
    obj.data.remesh_voxel_size = voxel_size
    bpy.ops.object.voxel_remesh()
    smooth = obj.modifiers.new('Soften the joins', 'SMOOTH')
    smooth.factor = 0.9
    smooth.iterations = smoothing
    bpy.ops.object.modifier_apply(modifier=smooth.name)
    if hip_blend:
        group_name = 'Pelvic join'
        hip = obj.vertex_groups.new(name=group_name)
        for vertex in obj.data.vertices:
            z = vertex.co.z
            weight = smoothstep(0.54, 0.76, z) * (1 - smoothstep(0.88, 1.14, z))
            if weight > 0:
                hip.add([vertex.index], weight, 'REPLACE')
        smooth_hip = obj.modifiers.new('Continuous hips', 'SMOOTH')
        smooth_hip.vertex_group = group_name
        smooth_hip.factor = 1.0
        smooth_hip.iterations = 42
        bpy.ops.object.modifier_apply(modifier=smooth_hip.name)
        obj.vertex_groups.remove(obj.vertex_groups[group_name])
    decimate = obj.modifiers.new('Reduce the sculpt mesh', 'DECIMATE')
    decimate.ratio = simplify
    bpy.ops.object.modifier_apply(modifier=decimate.name)
    obj.data.materials.clear()
    obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def build_body(clay):
    torso = loft('Torso form', [
        (0.673, 0, 0.003, 0.054, 0.080, 1.0),
        (0.722, 0, 0.000, 0.253, 0.176, 0.90),
        (0.798, 0, 0.008, 0.384, 0.229, 0.92),
        (0.884, 0, 0.009, 0.416, 0.253, 0.94),
        (1.045, 0, 0.004, 0.414, 0.265, 0.97),
        (1.236, 0, 0.007, 0.394, 0.254, 0.99),
        (1.432, 0, 0.013, 0.361, 0.224, 1.0),
        (1.611, 0, 0.027, 0.312, 0.195, 1.0),
        (1.733, 0, 0.031, 0.263, 0.178, 1.0),
        (1.804, 0, 0.028, 0.245, 0.171, 1.0),
        (1.879, 0, 0.025, 0.224, 0.156, 1.0),
        (1.928, 0, 0.022, 0.163, 0.121, 1.0),
        (1.941, 0, 0.022, 0.013, 0.015, 1.0),
    ], clay)
    legs = []
    profile = [
        (0.003, 0.244, -0.118, 0.166, 0.222, 0.74),
        (0.011, 0.244, -0.124, 0.198, 0.258, 0.78),
        (0.027, 0.244, -0.127, 0.220, 0.283, 0.81),
        (0.054, 0.244, -0.132, 0.230, 0.297, 0.82),
        (0.091, 0.243, -0.130, 0.232, 0.296, 0.84),
        (0.150, 0.241, -0.116, 0.228, 0.278, 0.86),
        (0.238, 0.240, -0.089, 0.220, 0.252, 0.89),
        (0.364, 0.235, -0.052, 0.212, 0.235, 0.92),
        (0.503, 0.233, -0.028, 0.201, 0.225, 0.92),
        (0.630, 0.231, -0.008, 0.195, 0.218, 0.91),
        (0.735, 0.219, 0.007, 0.199, 0.208, 0.92),
        (0.827, 0.205, 0.011, 0.197, 0.197, 0.98),
        (0.942, 0.188, 0.013, 0.166, 0.180, 1.0),
        (1.012, 0.173, 0.013, 0.014, 0.015, 1.0),
    ]
    for side in (-1, 1):
        legs.append(loft('Leg form', [
            (z, side*x, y, rx, ry, p) for z, x, y, rx, ry, p in profile
        ], clay))
    body = blend_forms('Body', [torso, *legs], clay, 0.015, 0.30, 7, hip_blend=True)
    floor = min(vertex.co.z for vertex in body.data.vertices)
    for vertex in body.data.vertices:
        vertex.co.z -= floor * (1 - smoothstep(0.02, 0.20, vertex.co.z))
    body.shape_key_add(name='Basis')
    breath = body.shape_key_add(name='Breath')
    breath.value = 0.0
    for base, shape in zip(body.data.vertices, breath.data):
        x, y, z = base.co
        torso_weight = smoothstep(0.91, 1.36, z) * (1 - smoothstep(1.58, 1.80, z))
        shape.co = (x * (1 + 0.021*torso_weight), y * (1 + 0.045*torso_weight), z)
    return body


def build_arm(side, clay):
    label = 'R' if side > 0 else 'L'
    # The asymmetric glove forms and small forward thumb are visible in the source.
    dz = 0.009 if side > 0 else 0.0
    profile = [
        (0.777, 0.529, -0.093, 0.004, 0.005, 1.0),
        (0.802, 0.533, -0.106, 0.068, 0.072, 1.0),
        (0.853, 0.534, -0.110, 0.111, 0.105, 1.0),
        (0.918, 0.533, -0.107, 0.133, 0.122, 1.0),
        (0.996, 0.527, -0.100, 0.135, 0.126, 1.0),
        (1.087, 0.503, -0.074, 0.133, 0.117, 1.0),
        (1.213, 0.469, -0.038, 0.120, 0.106, 1.0),
        (1.341, 0.423, -0.005, 0.112, 0.106, 1.0),
        (1.477, 0.373, 0.011, 0.107, 0.112, 1.0),
        (1.600, 0.322, 0.023, 0.099, 0.110, 1.0),
        (1.692, 0.285, 0.027, 0.085, 0.093, 1.0),
        (1.756, 0.263, 0.027, 0.056, 0.065, 1.0),
        (1.790, 0.256, 0.025, 0.005, 0.006, 1.0),
    ]
    arm = loft('Arm form', [(z + dz, side*x, y, rx, ry, p) for z, x, y, rx, ry, p in profile], clay)
    thumb = loft('Thumb form', [
        (0.972 + dz, side*0.420, -0.173, 0.004, 0.004, 1.0),
        (0.992 + dz, side*0.416, -0.179, 0.033, 0.032, 1.0),
        (1.031 + dz, side*0.418, -0.171, 0.052, 0.056, 1.0),
        (1.085 + dz, side*0.433, -0.137, 0.058, 0.062, 1.0),
        (1.143 + dz, side*0.454, -0.103, 0.043, 0.043, 1.0),
        (1.187 + dz, side*0.460, -0.076, 0.005, 0.005, 1.0),
    ], clay, 32)
    return blend_forms(f'Arm.{label}', [arm, thumb], clay, 0.0105, 0.43, 6)


def head_section(nz):
    radius = math.sqrt(max(0.0, 1 - nz*nz)) * (1 + 0.016*nz - 0.014*nz*nz)
    center_y = 0.005 - 0.024 * max(-nz, 0.0)
    return radius, center_y


def head_front(x, z):
    radius, cy = head_section((z - HEAD_CENTER) / HEAD_HEIGHT)
    ellipse_x = x / (HEAD_WIDTH * radius)
    return cy - HEAD_DEPTH * radius * math.sqrt(max(0.0, 1 - ellipse_x*ellipse_x))


def build_head(clay):
    segments, rings = 96, 72
    vertices = [(0, head_section(-1)[1], HEAD_CENTER - HEAD_HEIGHT)]
    for i in range(1, rings):
        nz = -math.cos(math.pi * i / rings)
        radius, cy = head_section(nz)
        for j in range(segments):
            phi = TAU * j / segments
            vertices.append((HEAD_WIDTH*radius*math.cos(phi),
                             cy + HEAD_DEPTH*radius*math.sin(phi), HEAD_CENTER + HEAD_HEIGHT*nz))
    top = len(vertices)
    vertices.append((0, head_section(1)[1], HEAD_CENTER + HEAD_HEIGHT))
    faces = [(0, 1 + (i + 1) % segments, 1 + i) for i in range(segments)]
    for offset in range(1, top - segments, segments):
        for i in range(segments):
            a, b = offset + i, offset + (i + 1) % segments
            faces.append((a, b, b + segments, a + segments))
    for i in range(segments):
        faces.append((top - segments + i, top - segments + (i + 1) % segments, top))
    return mesh_object('Head clay', vertices, faces, clay)


def face_patch(name, cx, cz, rx, rz, material, mouth=False):
    segments, rings = 56, 8
    params = [(0.0, 0.0)]
    for j in range(1, rings + 1):
        for i in range(segments):
            params.append((j / rings, TAU * i / segments))

    def points(width, height, offset_z=0):
        for radius, angle in params:
            cos, sin = math.cos(angle), math.sin(angle)
            power = 0.78 if mouth else 1.0
            x = cx + width*radius*math.copysign(abs(cos)**power, cos)
            z = cz + offset_z + height*radius*math.copysign(abs(sin)**power, sin)
            inset = 0.0015 + (0.003 if mouth else 0.0055) * (1 - radius**7)
            yield (x, head_front(x, z) - inset, z)

    faces = [(0, 1 + i, 1 + (i + 1) % segments) for i in range(segments)]
    for ring in range(rings - 1):
        offset = 1 + ring*segments
        for i in range(segments):
            a, b = offset + i, offset + (i + 1) % segments
            faces.append((a, a + segments, b + segments, b))
    patch = mesh_object(name, list(points(rx, rz)), faces, material)
    if mouth:
        patch.shape_key_add(name='Basis')
        for key, width, height, dz in [('MouthNarrow', 0.030, 0.0145, 0.002),
                                        ('MouthRound', 0.031, 0.036, -0.003)]:
            shape = patch.shape_key_add(name=key)
            for vertex, position in zip(shape.data, points(width, height, dz)):
                vertex.co = position
            shape.value = 0.0
    return patch


def build_rig():
    skeleton = bpy.data.armatures.new('Clay rig')
    rig = bpy.data.objects.new('Character rig', skeleton)
    bpy.context.collection.objects.link(rig)
    rig.show_in_front = True
    skeleton.display_type = 'STICK'
    activate([rig], rig)
    bpy.ops.object.mode_set(mode='EDIT')
    joints = [
        ('Root', (0, 0, 0.004), 0.19, None),
        ('Pelvis', (0, 0.002, 0.76), 0.27, 'Root'),
        ('Chest', (0, 0.018, 1.29), 0.36, 'Pelvis'),
        ('Head', HEAD_PIVOT, 0.43, 'Chest'),
    ]
    for side, label in [(-1, 'L'), (1, 'R')]:
        joints.extend([
            (f'Shoulder.{label}', (side*0.270, 0.026, 1.720), 0.16, 'Chest'),
            (f'Wrist.{label}', (side*0.488, -0.055, 1.175), 0.13, f'Shoulder.{label}'),
        ])
    for name, position, length, parent in joints:
        bone = skeleton.edit_bones.new(name)
        bone.head = position
        bone.tail = Vector(position) + Vector((0, 0, length))
        bone.roll = 0
        if parent:
            bone.parent = skeleton.edit_bones[parent]
    bpy.ops.object.mode_set(mode='OBJECT')
    return rig


def skin(obj, rig, weights_for_vertex):
    obj.parent = rig
    groups = {}
    for vertex in obj.data.vertices:
        for name, value in weights_for_vertex(vertex.co).items():
            if value <= 0:
                continue
            if name not in groups:
                groups[name] = obj.vertex_groups.new(name=name)
            groups[name].add([vertex.index], value, 'REPLACE')
    modifier = obj.modifiers.new('Clay rig', 'ARMATURE')
    modifier.object = rig


def body_weights(point):
    z = point.z
    above_feet = smoothstep(0.27, 1.02, z)
    chest = smoothstep(1.04, 1.66, z)
    return {'Root': 1 - above_feet, 'Pelvis': above_feet*(1 - chest), 'Chest': chest}


def arm_weights(point, label):
    hand = 1 - smoothstep(0.96, 1.38, point.z)
    chest = smoothstep(1.51, 1.80, point.z) * 0.40
    return {f'Shoulder.{label}': 1 - hand - chest, f'Wrist.{label}': hand, 'Chest': chest}


def setup_preview():
    scene = bpy.context.scene
    bpy.ops.mesh.primitive_plane_add(size=140, location=(0, 0, -0.002))
    ground = bpy.context.object
    ground.name = 'Studio floor (preview)'
    ground.data.materials.append(gray_material('Studio floor', 0.705, 0.285, 0.49))

    data = bpy.data.lights.new('Soft key', 'SUN')
    data.energy = 2.14
    data.angle = math.radians(14)
    key = bpy.data.objects.new(data.name, data)
    scene.collection.objects.link(key)
    key.rotation_euler = Vector((0.43, 0.54, -1)).to_track_quat('-Z', 'Y').to_euler()
    fill_data = bpy.data.lights.new('Front bounce', 'SUN')
    fill_data.energy = 0.50
    fill_data.angle = math.radians(60)
    fill = bpy.data.objects.new(fill_data.name, fill_data)
    scene.collection.objects.link(fill)
    fill.rotation_euler = Vector((0.08, 1, -0.20)).to_track_quat('-Z', 'Y').to_euler()
    world = bpy.data.worlds.new('Neutral fill')
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs[0].default_value = (0.33, 0.33, 0.33, 1)
    scene.world = world

    data = bpy.data.cameras.new('Reference camera')
    data.type = 'ORTHO'
    data.sensor_fit = 'VERTICAL'
    data.ortho_scale = FRAME_HEIGHT
    camera = bpy.data.objects.new(data.name, data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    orient_camera(camera, 0.0)

    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 40
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 1080
    scene.render.resolution_y = 1350
    scene.render.resolution_percentage = 50
    scene.render.image_settings.file_format = 'PNG'
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.render.fps = 30
    return camera


def orient_camera(camera, angle):
    aim = Vector((FRAME_CENTER_X, 0, FRAME_CENTER_Z))
    direction = Vector((math.sin(angle)*math.cos(ELEVATION),
                        -math.cos(angle)*math.cos(ELEVATION), math.sin(ELEVATION)))
    camera.location = aim + direction*9
    camera.rotation_euler = (-direction).to_track_quat('-Z', 'Y').to_euler()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview-dir', type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
    bpy.ops.wm.read_factory_settings(use_empty=True)

    clay = gray_material('Matte gray clay', 0.499, 0.87)
    eye = gray_material('Graphite eyes', 0.184, 0.83, 0.17)
    mouth_material = gray_material('Graphite mouth', 0.145, 0.93, 0.12)
    body = build_body(clay)
    left, right = build_arm(-1, clay), build_arm(1, clay)
    head = build_head(clay)
    eye_left = face_patch('Eye.L', -0.242, 2.241, 0.079, 0.093, eye)
    eye_right = face_patch('Eye.R', 0.242, 2.241, 0.080, 0.093, eye)
    mouth = face_patch('Mouth', 0.0, 2.027, 0.0415, 0.0275, mouth_material, mouth=True)
    rig = build_rig()
    skin(body, rig, body_weights)
    skin(left, rig, lambda point: arm_weights(point, 'L'))
    skin(right, rig, lambda point: arm_weights(point, 'R'))
    for obj in (head, eye_left, eye_right, mouth):
        skin(obj, rig, lambda point: {'Head': 1.0})

    camera = setup_preview()
    meshes = [body, left, right, head, eye_left, eye_right, mouth]
    activate([rig, *meshes], rig)
    assets = ROOT / 'assets'
    assets.mkdir(exist_ok=True)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(assets / 'character.blend'), compress=True)
    bpy.ops.export_scene.gltf(filepath=str(assets / 'character.glb'), export_format='GLB',
                              use_selection=True, export_animations=False, export_yup=True)
    print(f'Character: {sum(len(mesh.data.vertices) for mesh in meshes)} vertices; {len(rig.data.bones)} bones')

    if args.preview_dir:
        args.preview_dir.mkdir(exist_ok=True, parents=True)
        for label, angle in [('front', 0.0), ('three-quarter', math.radians(53))]:
            orient_camera(camera, angle)
            bpy.context.scene.render.filepath = str(args.preview_dir / f'{label}.png')
            bpy.ops.render.render(write_still=True)
        orient_camera(camera, 0.0)
        bpy.data.objects['Studio floor (preview)'].hide_render = True
        bpy.context.scene.render.film_transparent = True
        bpy.context.scene.render.image_settings.color_mode = 'RGBA'
        bpy.context.scene.render.filepath = str(args.preview_dir / 'matte.png')
        bpy.ops.render.render(write_still=True)


if __name__ == '__main__':
    main()
