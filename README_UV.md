# EgoBlur Demo

This repository contains demo of [EgoBlur models](https://www.projectaria.com/tools/egoblur) with visualizations.

## Installation

### Quick Start with UV (Recommended)

EgoBlur now supports installation with [UV](https://github.com/astral-sh/uv), a fast Python package installer and resolver. UV provides faster dependency resolution and better reproducibility compared to conda.

#### Prerequisites

1. **Install UV** (if not already installed):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **System Dependencies**:
   - **Linux**: `ffmpeg` (will be installed automatically by our script)
   - **macOS**: `ffmpeg` via Homebrew
   - **Windows**: `ffmpeg` via vcpkg or manual installation

#### Installation Options

**Option 1: Automatic Installation (Recommended)**

```bash
# Clone the repository
git clone https://github.com/facebookresearch/EgoBlur.git
cd EgoBlur

# Run the installation script (auto-detects CUDA)
./install.sh
```

**Option 2: Manual Installation**

```bash
# Clone the repository
git clone https://github.com/facebookresearch/EgoBlur.git
cd EgoBlur

# Install dependencies
uv sync
```

**Option 3: Specific CUDA Version**

```bash
# CUDA 11.8
./install.sh --cuda-version 11.8

# CUDA 12.1
./install.sh --cuda-version 12.1

# CUDA 12.4
./install.sh --cuda-version 12.4

# CPU-only
./install.sh --cpu
```

#### Verify Installation

After installation, verify PyTorch and CUDA support:

```bash
uv run python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

You can also run the test script to verify everything is working:

```bash
uv run python test_installation.py
```

**Note**: This version uses MoviePy 2.x which has updated import syntax. The demo script has been updated to use `from moviepy import ImageSequenceClip` instead of the older `from moviepy.editor import ImageSequenceClip`.

### Legacy Conda Installation

If you prefer to use conda, you can still use the original installation method. See the original [README.md](README.md) for conda installation instructions.

## Getting Started

First download the zipped models from given links. Then the models can be used as input/s to CLI.

| Model         | Download link                                                 |
| ------------- | ------------------------------------------------------------- |
| ego_blur_face | [ego_blur_website](https://www.projectaria.com/tools/egoblur) |
| ego_blur_lp   | [ego_blur_website](https://www.projectaria.com/tools/egoblur) |

### CLI options

A brief description of CLI args:

`--face_model_path` use this argument to provide absolute EgoBlur face model file path. You MUST provide either `--face_model_path` or `--lp_model_path` or both. If none is provided code will throw a `ValueError`.

`--face_model_score_threshold` use this argument to provide face model score threshold to filter out low confidence face detections. The values must be between 0.0 and 1.0, if not provided this defaults to 0.1.

`--lp_model_path` use this argument to provide absolute EgoBlur license plate file path. You MUST provide either `--face_model_path` or `--lp_model_path` or both. If none is provided code will throw a `ValueError`.

`--lp_model_score_threshold` use this argument to provide license plate model score threshold to filter out low confidence license plate detections. The values must be between 0.0 and 1.0, if not provided this defaults to 0.1.

`--nms_iou_threshold` use this argument to provide NMS iou threshold to filter out low confidence overlapping boxes. The values must be between 0.0 and 1.0, if not provided this defaults to 0.3.

`--scale_factor_detections` use this argument to provide scale detections by the given factor to allow blurring more area. The values can only be positive real numbers eg: 0.9(values < 1) would mean scaling DOWN the predicted blurred region by 10%, whereas as 1.1(values > 1) would mean scaling UP the predicted blurred region by 10%.

`--input_image_path` use this argument to provide absolute path for the given image on which we want to make detections and perform blurring. You MUST provide either `--input_image_path` or `--input_video_path` or both. If none is provided code will throw a `ValueError`.

`--output_image_path` use this argument to provide absolute path where we want to store the blurred image. You MUST provide `--output_image_path` with `--input_image_path` otherwise code will throw `ValueError`.

`--input_video_path` use this argument to provide absolute path for the given video on which we want to make detections and perform blurring. You MUST provide either `--input_image_path` or `--input_video_path` or both. If none is provided code will throw a `ValueError`.

`--output_video_path` use this argument to provide absolute path where we want to store the blurred video. You MUST provide `--output_video_path` with `--input_video_path` otherwise code will throw `ValueError`.

`--output_video_fps` use this argument to provide FPS for the output video. The values must be positive integers, if not provided this defaults to 30.

### CLI command example

Download the git repo locally and run following commands.
Please note that these commands assumes that you have a created a folder `/home/${USER}/ego_blur_assets/` where you have extracted the zipped models and have test image in the form of `test_image.jpg` and a test video in the form of `test_video.mp4`.

```
# For UV installation (recommended)
uv run python script/demo_ego_blur.py --help
```

#### demo command for face blurring on the demo_assets image

```
uv run python script/demo_ego_blur.py --face_model_path /home/${USER}/ego_blur_assets/ego_blur_face.jit --input_image_path demo_assets/test_image.jpg --output_image_path /home/${USER}/ego_blur_assets/test_image_output.jpg
```

#### demo command for face blurring on an image using default arguments

```
uv run python script/demo_ego_blur.py --face_model_path /home/${USER}/ego_blur_assets/ego_blur_face.jit --input_image_path /home/${USER}/ego_blur_assets/test_image.jpg --output_image_path /home/${USER}/ego_blur_assets/test_image_output.jpg
```

#### demo command for face blurring on an image

```
uv run python script/demo_ego_blur.py --face_model_path /home/${USER}/ego_blur_assets/ego_blur_face.jit --input_image_path /home/${USER}/ego_blur_assets/test_image.jpg --output_image_path /home/${USER}/ego_blur_assets/test_image_output.jpg --face_model_score_threshold 0.9 --nms_iou_threshold 0.3 --scale_factor_detections 1.15
```

#### demo command for license plate blurring on an image

```
uv run python script/demo_ego_blur.py --lp_model_path /home/${USER}/ego_blur_assets/ego_blur_lp.jit --input_image_path /home/${USER}/ego_blur_assets/test_image.jpg --output_image_path /home/${USER}/ego_blur_assets/test_image_output.jpg --lp_model_score_threshold 0.9 --nms_iou_threshold 0.3 --scale_factor_detections 1
```

#### demo command for face blurring and license plate blurring on an input image and video

```
uv run python script/demo_ego_blur.py --face_model_path /home/${USER}/ego_blur_assets/ego_blur_face.jit --lp_model_path /home/${USER}/ego_blur_assets/ego_blur_lp.jit --input_image_path /home/${USER}/ego_blur_assets/test_image.jpg --output_image_path /home/${USER}/ego_blur_assets/test_image_output.jpg  --input_video_path /home/${USER}/ego_blur_assets/test_video.mp4 --output_video_path /home/${USER}/ego_blur_assets/test_video_output.mp4 --face_model_score_threshold 0.9 --lp_model_score_threshold 0.9 --nms_iou_threshold 0.3 --scale_factor_detections 1 --output_video_fps 20
```

#### Alternative: Using the CLI command (if installed as package)

```
# After installation, you can also use the CLI command
uv run egoblur-demo --help
uv run egoblur-demo --face_model_path /path/to/model.jit --input_image_path /path/to/image.jpg --output_image_path /path/to/output.jpg
```

## License

The model is licensed under the [Apache 2.0 license](LICENSE).

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Citing EgoBlur

If you use EgoBlur in your research, please use the following BibTeX entry.

```
@misc{raina2023egoblur,
      title={EgoBlur: Responsible Innovation in Aria},
      author={Nikhil Raina and Guruprasad Somasundaram and Kang Zheng and Sagar Miglani and Steve Saarinen and Jeff Meissner and Mark Schwesinger and Luis Pesqueira and Ishita Prasad and Edward Miller and Prince Gupta and Mingfei Yan and Richard Newcombe and Carl Ren and Omkar M Parkhi},
      year={2023},
      eprint={2308.13093},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Migration from Conda

This repository now supports both conda and UV installation methods:

- **UV (Recommended)**: Faster, more reliable, better cross-platform support
- **Conda (Legacy)**: Original installation method, see [README.md](README.md)

### Benefits of UV over Conda:

- ⚡ **Faster installation**: Rust-based resolver is significantly faster
- 🔒 **Better reproducibility**: Lock files and precise version management
- 🌍 **Cross-platform**: Works seamlessly on macOS ARM64, Linux x86_64, Windows
- 📦 **Simplified workflow**: Single `uv sync` command installs everything
- 🔄 **Modern dependencies**: Automatically uses latest compatible versions
- 🛠️ **Better tooling**: Integrated with modern Python development tools

### Quick Migration Guide:

1. Install UV: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Clone repository: `git clone https://github.com/facebookresearch/EgoBlur.git`
3. Install: `cd EgoBlur && uv sync`
4. Run: `uv run python script/demo_ego_blur.py --help`
