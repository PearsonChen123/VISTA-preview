# Stereo Depth Pipeline

One command runs from raw images to filtered depth maps. Point-cloud, fusion,
mesh, and multi-model comparison tools have moved to `_archive/`.

```
raw images
   |
   |-- COLMAP ---------------------------------------------
   |   colmap       SfM and undistortion
   |   transforms   undistorted model -> transforms.json
   |
   |-- nerfstudio -----------------------------------------
   |   train        train a radiance field
   |   export       export training-view camera poses
   |   shift        generate right-camera poses
   |   campath      convert to nerfstudio camera_path
   |   render       render left and right views
   |
   |-- stereo matching ------------------------------------
   |   rotate       rotate into the stereo convention
   |   intrinsic    generate K.txt and a metric baseline
   |   stereo       run FoundationStereo
   |   depth        restore orientation and save 16-bit PNG
   |   filter       reject inconsistent depth with multi-view checks
   v
depth maps (.npy in meters / 16-bit .png)
```

## Usage

```bash
./run_pipeline.sh -c config.json
./run_pipeline.sh -c config.json --from render
./run_pipeline.sh -c config.json --only stereo,depth,filter
./run_pipeline.sh -c config.json --show
```

All parameters live in `config.json`; the CLI provides flow control and common
overrides such as `--shift`, `--shift-direction`, `--vis`, and `--clean`.

```json
{
  "project":    { "root": "/path/to/scene" },
  "colmap":     { "enabled": true, "image_dir": "images", "matcher": "exhaustive" },
  "nerfstudio": { "method": "nerfacto", "max_num_iterations": 30000 },
  "stereo":     { "shift_mode": "pixels", "shift_pixels": 0.1, "direction": "up" },
  "filter":     { "enabled": true, "max_depth_error": 0.01, "use_ncc": true },
  "output":     { "depth_png": true, "clean_intermediate": false, "visualize": false }
}
```

Unspecified values use defaults from `lib/load_config.py`. Relative paths resolve
against the configuration file. Set `colmap.enabled` to false when
`transforms.json` already exists. For an existing trained model, start at
`export` or set `nerfstudio.config_path`.

## COLMAP

The `colmap` stage wraps:

```
feature_extractor -> exhaustive_matcher -> mapper -> image_undistorter
```

Existing stage products are skipped automatically. Use the sequential matcher
for extracted video frames.

Undistortion matters because OPENCV and RADIAL mapper models contain distortion
parameters that every training, rendering, and stereo stage would otherwise
need to preserve. `image_undistorter` produces a PINHOLE model described only by
fx/fy/cx/cy. The converter reads COLMAP binary files directly.

COLMAP stores world-to-camera poses in OpenCV coordinates (x-right, y-down,
z-forward); nerfstudio uses camera-to-world OpenGL coordinates (x-right, y-up,
z-back):

```
transform_matrix = inv(w2c) @ diag(1, -1, -1, 1)
```

A known-pose binary model was validated with `colmap model_analyzer` and converted
back with a maximum error of 1.3e-15.

## Layout

```
scripts/
├── run_pipeline.sh              Main entry point
├── config.json                  Configuration
├── lib/
│   ├── load_config.py           config.json -> shell variables
│   ├── run_colmap.py            COLMAP SfM and undistortion
│   ├── colmap_model.py          COLMAP binary model reader
│   ├── colmap_to_transforms.py  transforms.json conversion
│   ├── export_poses.py          Camera pose export
│   ├── stereo_shift.py          Left poses -> right poses
│   ├── make_camera_path.py      nerfstudio camera_path conversion
│   ├── rotate_images.py         Stereo image rotation
│   ├── make_intrinsics.py       K.txt generation
│   ├── depth_postprocess.py     Orientation restoration and PNG output
│   ├── filter_depth.py          Multi-view consistency filtering
│   └── common.py                Shared rotations, intrinsics, and K.txt I/O
└── _archive/                    Legacy tools
```

Scene products default to `<project.root>/stereo_depth/`:

```
stereo_depth/
├── cam_path/        poses, camera paths, and K.txt
├── render/          original left/right renders
├── rotated/         network-ready rotated images
├── raw_depth/       raw FoundationStereo output
├── depth/           metric .npy and 16-bit .png depth
├── depth_filtered/  filtered depth with unreliable pixels set to zero
└── vis_filter/      optional filter visualizations
```

For 30 frames at 640x480, products use about 108 MB: 36 MB each for raw and final
float32 depth, 19 MB for rotated images, 16 MB for renders, and 2.9 MB for depth
PNG files. `clean_intermediate` removes `rotated/` and `raw_depth/`, reducing the
measured total from 108 MB to 55 MB. After cleanup, rerunning `depth` also requires
rerunning `stereo`.

## Depth Filtering

FoundationStereo emits dense depth without per-pixel reliability. The filter
cross-validates every reference pixel against source views using:

| Check | Default |
|---|---|
| Round-trip reprojection error | 2.0 px |
| Relative depth error | 0.01 |
| Minimum triangulation angle | 3 degrees |
| NCC photometric consistency | 0.3 |
| Required consistent views | 2 |
| Source views | 8 |

Round-trip projection follows reference -> source -> reference and uses the
noninteger projected source coordinate when constructing the reverse ray.
Windowed NCC is computed with box filters:

```
NCC = (E[rw] - E[r]E[w]) / sqrt(Var[r] * Var[w])
```

NCC abstains when reference-window texture standard deviation is below 0.02.
Textureless regions do not imply bad depth, and forcing an NCC decision reduced
measured retention from 50% to 13.6%.

Filtering has no cross-pixel propagation dependency. A 480x640 image with eight
source views creates 2.46 million independent units. The measured runtime is
0.35 seconds for 30 views, or 11.6 ms/view.

On a 30-view 640x480 synthetic scene with ground-truth depth:

| | Retained | Median relative error | delta<5% | delta<10% |
|---|---|---|---|---|
| Before | 100% | 1.08% | 82.6% | 91.5% |
| After | 50.1% | **0.47%** | **98.7%** | **99.5%** |

The default is deliberately conservative: it detects 96.2% of pixels with more
than 5% error and leaves retained depth at 98.7% purity.

## Baseline

The default baseline targets disparity as a fraction of image width:

```json
"stereo": { "shift_mode": "pixels", "shift_pixels": 0.1 }
```

Disparity varies with depth as `d = fx*B/Z`, so the target is anchored at a scene
reference depth estimated from visible COLMAP sparse points. The default uses
the 25th percentile rather than the median to keep foreground disparity within
the stereo network's range. An explicit `stereo.reference_depth` can replace
point-cloud estimation.

```
baseline_metric = shift_pixels * W * Z_ref / fx
shift_norm      = baseline_metric * scale
                  ... translate poses in normalized space ...
baseline        = shift_norm / scale
```

Legacy direct baselines remain available with
`"shift_mode": "baseline", "shift": 0.2`.

## Rotation And Direction

Stereo networks require horizontal disparity. Images, intrinsics, and output
depth must use consistent rotations, defined once in `lib/common.py`.
`direction` describes right-image content motion relative to the left image:

| Direction | Image/intrinsic rotation | Restore depth | Legacy |
|---|---|---|---|
| `up` | 90 degrees counterclockwise | 90 degrees clockwise | `y` |
| `down` | 90 degrees clockwise | 90 degrees counterclockwise | `-y` |
| `left` | none | none | `-x` |
| `right` | 180 degrees | 180 degrees | `x` |

Automatic direction selection projects trajectory tangents into each camera's
coordinates and chooses the majority dominant axis and sign. Following the
observed trajectory keeps virtual views near well-trained NeRF viewpoints.
Disable it with `"auto_direction": false` or `--no-auto-direction`.

## Environments

The pipeline invokes absolute interpreter paths rather than activating conda
environments:

| Stages | Environment |
|---|---|
| `export`, `render` | `nerfstudio_sm120` |
| All others | `nevstereo` |

FoundationStereo code and weights live under `third_party/FoundationStereo/` and
`models/foundation_stereo/`. On the RTX 5090 it runs in `nevstereo` with
torch 2.8.0+cu128. When flash-attn is unavailable, the implementation falls back
to mathematically equivalent PyTorch scaled-dot-product attention. A synthetic
24-pixel shift produced 23.99 px estimated disparity, validating the fallback.
