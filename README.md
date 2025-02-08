# LatentSync with SuperResolution Enhancement

This repository is a modified version of [LatentSync by ByteDance](https://github.com/bytedance/LatentSync) and introduces optional superresolution capabilities using either [GFPGAN](https://github.com/TencentARC/GFPGAN) or [CodeFormer](https://github.com/sczhou/CodeFormer). The enhancements were implemented to tackle the issue where the generated subframe used for lipsynced video output is of lower resolution than the input frame—resulting in subpar video quality.

## Features

- **SuperResolution on Generated Subframes:**  
  A new parameter `--superres` has been added to the inference pipeline. It accepts three options:
  - `GFPGAN`: Applies GFPGAN based superresolution on the generated subframe.
  - `CodeFormer`: Applies CodeFormer based superresolution on the generated subframe.
  - `none`: Disables superresolution (default).

- **Adaptive Resolution Adjustment:**  
  The code dynamically checks the resolution ratio between the input frame and the generated subframe. If the generated portion is of lower resolution, it computes the upscale factor and applies the selected superresolution framework exclusively on the generated part.

- **Efficient Integration:**  
  Superresolution is applied only when necessary and solely on the subframe, ensuring that the performance overhead is minimized.

## How It Works

1. **Inference Pipeline:**  
   The script `inference.py` processes the input video frame by frame. For each frame, after generating the lipsynced subframe, the code checks if the subframe's resolution is lower than the original frame's resolution.

2. **Resolution Ratio Calculation:**  
   If the generated subframe is detected to be of lower quality, the upscale ratio is calculated based on the input and generated dimensions.

3. **Selective SuperResolution:**  
   Depending on the `--superres` argument provided, the selected framework (GFPGAN or CodeFormer) is used to upscale only the generated region, enhancing the final output quality without affecting the rest of the frame.

4. **Output Video Generation:**  
   The enhanced frames are then compiled to produce the final video output.

## Getting Started

### Prerequisites

- Python 3.8 or later
- PyTorch
- OpenCV
- NumPy
- Libraries for GFPGAN and CodeFormer (ensure they are installed and properly set up if you plan to use these features)
- Additional dependencies as per the original LatentSync requirements

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Niketan77/videodubber.ai-Project.git
   cd videodubber.ai-Project
