"""Create the versioned, self-contained K2Lab depth-authoring template.

Run with:
    blender --background --python blender/create_template.py
"""

from __future__ import annotations

from pathlib import Path

import bpy
from mathutils import Vector


TEMPLATE_VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "template" / "k2lab_depth_template_v1.blend"


def collection(name: str) -> bpy.types.Collection:
    result = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(result)
    return result


def move_to(obj: bpy.types.Object, target: bpy.types.Collection) -> None:
    for source in tuple(obj.users_collection):
        source.objects.unlink(obj)
    target.objects.link(obj)


def cube(
    name: str,
    location: tuple[float, float, float],
    scale: tuple[float, float, float],
    target: bpy.types.Collection,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    move_to(obj, target)
    return obj


def segment(
    name: str,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    radius: float,
    bone_name: str,
    rig: bpy.types.Object,
    target: bpy.types.Collection,
) -> bpy.types.Object:
    a, b = Vector(start), Vector(end)
    direction = b - a
    bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=12, location=(a + b) / 2)
    obj = bpy.context.object
    obj.name = name
    obj.scale = (radius, radius, direction.length / 2)
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = direction.to_track_quat("Z", "Y")
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    group = obj.vertex_groups.new(name=bone_name)
    group.add(range(len(obj.data.vertices)), 1.0, "REPLACE")
    modifier = obj.modifiers.new(name="K2Lab mannequin rig", type="ARMATURE")
    modifier.object = rig
    obj.parent = rig
    move_to(obj, target)
    return obj


def create_mannequin(target: bpy.types.Collection) -> None:
    armature = bpy.data.armatures.new("mannequin_001_rig")
    rig = bpy.data.objects.new("mannequin_001", armature)
    target.objects.link(rig)
    rig.show_in_front = True
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bones = {
        "root": ((0, 0, 0), (0, 0, 0.9), None),
        "spine": ((0, 0, 0.9), (0, 0, 2.0), "root"),
        "neck": ((0, 0, 2.0), (0, 0, 2.25), "spine"),
        "head": ((0, 0, 2.25), (0, 0, 2.75), "neck"),
        "upper_arm.L": ((0, 0, 1.9), (-0.75, 0, 1.75), "spine"),
        "forearm.L": ((-0.75, 0, 1.75), (-1.35, 0, 1.45), "upper_arm.L"),
        "upper_arm.R": ((0, 0, 1.9), (0.75, 0, 1.75), "spine"),
        "forearm.R": ((0.75, 0, 1.75), (1.35, 0, 1.45), "upper_arm.R"),
        "thigh.L": ((-0.27, 0, 0.9), (-0.3, 0, -0.15), "root"),
        "shin.L": ((-0.3, 0, -0.15), (-0.3, 0, -1.1), "thigh.L"),
        "thigh.R": ((0.27, 0, 0.9), (0.3, 0, -0.15), "root"),
        "shin.R": ((0.3, 0, -0.15), (0.3, 0, -1.1), "thigh.R"),
    }
    edit_bones = {}
    for name, (head, tail, parent) in bones.items():
        bone = armature.edit_bones.new(name)
        bone.head, bone.tail = head, tail
        if parent:
            bone.parent = edit_bones[parent]
        edit_bones[name] = bone
    bpy.ops.object.mode_set(mode="OBJECT")
    rig.location = (-0.8, 0.2, 1.1)
    for name, (start, end, _parent) in bones.items():
        radius = 0.31 if name in {"root", "spine"} else 0.17
        if name == "head":
            radius = 0.29
        segment(
            f"mannequin_001_{name.replace('.', '_')}",
            start,
            end,
            radius,
            name,
            rig,
            target,
        )
    rig["k2lab_kind"] = "poseable_mannequin"
    rig["k2lab_license"] = "Original procedural K2Lab asset; CC0-1.0"


def look_at(obj: bpy.types.Object, point: tuple[float, float, float]) -> None:
    obj.rotation_euler = (Vector(point) - obj.location).to_track_quat("-Z", "Y").to_euler()


def main() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for existing in tuple(bpy.data.collections):
        bpy.data.collections.remove(existing)
    characters = collection("Characters")
    furniture = collection("Furniture")
    environment = collection("Environment")
    collection("Lights")
    cameras = collection("Cameras")
    calibration = collection("Calibration")

    create_mannequin(characters)
    cube("floor_001", (0, 0, -0.08), (4.5, 4.5, 0.08), environment)
    cube("wall_back_001", (0, 3.5, 2.0), (4.5, 0.08, 2.0), environment)
    cube("bed_001", (1.4, 1.0, 0.42), (1.2, 2.0, 0.35), furniture)
    cube("bed_001_headboard", (1.4, 2.8, 1.05), (1.2, 0.12, 0.75), furniture)
    cube("chair_001_seat", (-2.0, 0.8, 0.75), (0.55, 0.55, 0.12), furniture)
    cube("chair_001_back", (-2.0, 1.3, 1.35), (0.55, 0.12, 0.7), furniture)
    for x in (-2.45, -1.55):
        for y in (0.4, 1.2):
            cube(f"chair_001_leg_{x}_{y}", (x, y, 0.35), (0.07, 0.07, 0.35), furniture)
    cube("table_001_top", (0, -1.9, 1.0), (1.3, 0.75, 0.12), furniture)
    for x in (-1.05, 1.05):
        for y in (-2.4, -1.4):
            cube(f"table_001_leg_{x}_{y}", (x, y, 0.5), (0.09, 0.09, 0.5), furniture)
    cube("sofa_001_seat", (2.6, -1.4, 0.65), (1.35, 0.65, 0.3), furniture)
    cube("sofa_001_back", (2.6, -0.85, 1.35), (1.35, 0.15, 0.75), furniture)

    # An asymmetric near-left/tall-right pair makes flips immediately visible.
    cube("orientation_near_left", (-2.7, -2.5, 0.45), (0.45, 0.45, 0.45), calibration)
    cube("orientation_far_right_tall", (2.8, 2.2, 1.25), (0.35, 0.35, 1.25), calibration)

    camera_data = bpy.data.cameras.new("K2Lab_Camera")
    camera = bpy.data.objects.new("K2Lab_Camera", camera_data)
    cameras.objects.link(camera)
    camera.location = (8.5, -11.5, 7.0)
    camera_data.lens = 48
    camera_data.sensor_width = 36
    camera_data.clip_start = 0.1
    camera_data.clip_end = 50.0
    look_at(camera, (0, 0.3, 1.0))
    scene = bpy.context.scene
    scene.camera = camera
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "BW"
    scene.render.image_settings.color_depth = "16"
    scene.render.film_transparent = False
    scene.world.color = (0.04, 0.04, 0.04)
    scene["k2lab_template_version"] = TEMPLATE_VERSION
    scene["k2lab_metric_scale"] = 1.0
    scene["k2lab_depth_convention"] = "near_white_far_black"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT))
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
