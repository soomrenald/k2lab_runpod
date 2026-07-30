# K2Lab Blender depth authoring

The checked-in template is generated entirely from `create_template.py`; it uses
no third-party geometry. The procedural mannequin and furniture are released
with this project under CC0-1.0 so scenes may be edited and redistributed.

Create or refresh the template:

```bash
blender --background --python blender/create_template.py
```

Export the active scene camera:

```bash
blender blender/template/k2lab_depth_template_v1.blend \
  --background --python blender/export_scene.py -- \
  --output reports/blender-reference --width 1024 --height 1024
```

The exporter writes a 16-bit grayscale depth image, 8-bit preview, 16-bit
object-ID image, camera and object metadata, a scene copy, and a checksum
manifest. Depth is normalized deterministically between the active camera clip
planes with near objects white and far objects black. Images use a top-left
origin and are not vertically flipped relative to K2Lab canvas coordinates.

The `Calibration` collection deliberately contains a near-left short object and
a far-right tall object. Keep it visible while checking a new Blender version or
camera setup; a mirrored import is immediately apparent.
