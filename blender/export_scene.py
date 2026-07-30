"""Export a deterministic K2Lab depth bundle from the active Blender camera."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import bpy
import numpy as np


BUNDLE_VERSION = 1


def arguments() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--no-scene-copy", action="store_true")
    return parser.parse_args(values)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def matrix_values(value) -> list[list[float]]:
    return [[float(item) for item in row] for row in value]


def configure_compositor(output: Path, near: float, far: float) -> None:
    scene = bpy.context.scene
    tree = getattr(scene, "node_tree", None)
    if tree is None:
        tree = scene.compositing_node_group
    if tree is None:
        tree = bpy.data.node_groups.new("K2Lab Depth Export", "CompositorNodeTree")
        scene.compositing_node_group = tree
    tree.nodes.clear()
    layers = tree.nodes.new("CompositorNodeRLayers")
    normalize = tree.nodes.new("ShaderNodeMapRange")
    normalize.inputs["From Min"].default_value = near
    normalize.inputs["From Max"].default_value = far
    normalize.inputs["To Min"].default_value = 1.0
    normalize.inputs["To Max"].default_value = 0.0
    normalize.clamp = True
    tree.links.new(layers.outputs["Depth"], normalize.inputs["Value"])

    depth = tree.nodes.new("CompositorNodeOutputFile")
    depth.directory = str(output)
    depth.file_name = "depth_16bit"
    depth_item = depth.file_output_items.new("FLOAT", "Depth")
    depth_item.override_node_format = True
    depth_item.format.file_format = "PNG"
    depth_item.format.color_mode = "BW"
    depth_item.format.color_depth = "16"
    tree.links.new(normalize.outputs[0], depth.inputs["Depth"])

    preview = tree.nodes.new("CompositorNodeOutputFile")
    preview.directory = str(output)
    preview.file_name = "depth_preview"
    preview_item = preview.file_output_items.new("FLOAT", "Depth")
    preview_item.override_node_format = True
    preview_item.format.file_format = "PNG"
    preview_item.format.color_mode = "BW"
    preview_item.format.color_depth = "8"
    tree.links.new(normalize.outputs[0], preview.inputs["Depth"])

    object_map = tree.nodes.new("ShaderNodeMath")
    object_map.operation = "MULTIPLY"
    object_map.inputs[1].default_value = 1.0 / 65535.0
    object_map.use_clamp = True
    tree.links.new(layers.outputs["Object Index"], object_map.inputs[0])
    ids = tree.nodes.new("CompositorNodeOutputFile")
    ids.directory = str(output)
    ids.file_name = "object_ids"
    ids_item = ids.file_output_items.new("FLOAT", "Object IDs")
    ids_item.override_node_format = True
    ids_item.format.file_format = "PNG"
    ids_item.format.color_mode = "BW"
    ids_item.format.color_depth = "16"
    tree.links.new(object_map.outputs[0], ids.inputs["Object IDs"])


def rendered_exr(output: Path, stem: str) -> Path:
    candidates = sorted(output.glob(f"{stem}*.exr"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one compositor output for {stem}, found {candidates}")
    return candidates[0]


def convert_exr_to_png(source: Path, destination: Path, bit_depth: str) -> Path:
    scene = bpy.context.scene
    image = bpy.data.images.load(str(source), check_existing=False)
    try:
        image.colorspace_settings.name = "Non-Color"
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "BW"
        scene.render.image_settings.color_depth = bit_depth
        image.save_render(str(destination), scene=scene)
    finally:
        bpy.data.images.remove(image)
    source.unlink()
    return destination


def export_object_masks(
    object_ids: Path,
    objects: list[bpy.types.Object],
    output: Path,
) -> dict[str, str]:
    masks = output / "masks"
    masks.mkdir(exist_ok=True)
    image = bpy.data.images.load(str(object_ids), check_existing=False)
    image.colorspace_settings.name = "Non-Color"
    pixels = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(pixels)
    values = pixels.reshape(-1, image.channels)[:, 0]
    checksums: dict[str, str] = {}
    try:
        for obj in objects:
            if obj.type != "MESH":
                continue
            target = obj.pass_index / 65535.0
            mask_values = np.isclose(
                values,
                target,
                rtol=0.0,
                atol=0.5 / 65535.0,
            ).astype(np.float32)
            rgba = np.repeat(mask_values[:, None], 4, axis=1)
            rgba[:, 3] = 1.0
            mask = bpy.data.images.new(
                f"K2Lab mask {obj.name}",
                width=image.size[0],
                height=image.size[1],
                alpha=False,
            )
            try:
                mask.pixels.foreach_set(rgba.reshape(-1))
                destination = masks / f"{obj.name}.png"
                scene = bpy.context.scene
                scene.render.image_settings.file_format = "PNG"
                scene.render.image_settings.color_mode = "BW"
                scene.render.image_settings.color_depth = "8"
                mask.save_render(str(destination), scene=scene)
                relative = destination.relative_to(output).as_posix()
                checksums[relative] = sha256(destination)
            finally:
                bpy.data.images.remove(mask)
    finally:
        bpy.data.images.remove(image)
    return checksums


def main() -> None:
    args = arguments()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    camera = scene.camera
    if camera is None or camera.type != "CAMERA":
        raise RuntimeError("K2Lab export requires one active camera")
    if args.width is not None:
        scene.render.resolution_x = args.width
    if args.height is not None:
        scene.render.resolution_y = args.height
    if scene.render.resolution_x <= 0 or scene.render.resolution_y <= 0:
        raise RuntimeError("render resolution must be positive")
    scene.render.resolution_percentage = 100
    # Cycles exposes deterministic Z and Object Index passes in both desktop and
    # headless Blender. One CPU sample is sufficient because only data passes
    # are exported.
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    scene.view_layers[0].use_pass_z = True
    scene.view_layers[0].use_pass_object_index = True
    export_objects = sorted(
        (obj for obj in scene.objects if obj.type in {"MESH", "ARMATURE"}),
        key=lambda obj: obj.name,
    )
    for index, obj in enumerate(export_objects, start=1):
        obj.pass_index = index
    configure_compositor(
        output,
        float(camera.data.clip_start),
        float(camera.data.clip_end),
    )
    bpy.ops.render.render()
    depth = convert_exr_to_png(
        rendered_exr(output, "depth_16bit"),
        output / "depth_16bit.png",
        "16",
    )
    preview = convert_exr_to_png(
        rendered_exr(output, "depth_preview"),
        output / "depth_preview.png",
        "8",
    )
    object_ids = convert_exr_to_png(
        rendered_exr(output, "object_ids"),
        output / "object_ids.png",
        "16",
    )
    mask_checksums = export_object_masks(object_ids, export_objects, output)

    camera_document = {
        "name": camera.name,
        "type": camera.data.type,
        "lens_mm": float(camera.data.lens),
        "sensor_width_mm": float(camera.data.sensor_width),
        "clip_start": float(camera.data.clip_start),
        "clip_end": float(camera.data.clip_end),
        "matrix_world": matrix_values(camera.matrix_world),
        "resolution": [scene.render.resolution_x, scene.render.resolution_y],
        "pixel_aspect": [scene.render.pixel_aspect_x, scene.render.pixel_aspect_y],
        "depth_convention": "near_white_far_black",
        "origin": "top_left",
        "vertical_flip": False,
    }
    objects_document = {
        "objects": [
            {
                "name": obj.name,
                "id": int(obj.pass_index),
                "type": obj.type,
                "collections": sorted(item.name for item in obj.users_collection),
                "matrix_world": matrix_values(obj.matrix_world),
                "location": [float(value) for value in obj.location],
                "rotation_euler": [float(value) for value in obj.rotation_euler],
                "scale": [float(value) for value in obj.scale],
            }
            for obj in export_objects
        ]
    }
    (output / "camera.json").write_text(
        json.dumps(camera_document, indent=2),
        encoding="utf-8",
    )
    (output / "objects.json").write_text(
        json.dumps(objects_document, indent=2),
        encoding="utf-8",
    )
    if not args.no_scene_copy:
        bpy.ops.wm.save_as_mainfile(filepath=str(output / "scene.blend"), copy=True)
    checked = [depth, preview, object_ids, output / "camera.json", output / "objects.json"]
    if (output / "scene.blend").is_file():
        checked.append(output / "scene.blend")
    manifest = {
        "format": "k2lab-blender-depth-bundle",
        "version": BUNDLE_VERSION,
        "template_version": str(scene.get("k2lab_template_version", "custom")),
        "blender_version": bpy.app.version_string,
        "exported_at": datetime.now(UTC).isoformat(),
        "scene_scale_meters": float(scene.unit_settings.scale_length or 1.0),
        "depth_image": depth.name,
        "depth_preview": preview.name,
        "object_ids": object_ids.name,
        "camera": "camera.json",
        "objects": "objects.json",
        "checksums": {path.name: sha256(path) for path in checked},
    }
    manifest["checksums"].update(mask_checksums)
    (output / "export.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
