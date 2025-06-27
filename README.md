# SceneEval

### SceneEval: Evaluating Semantic Coherence in Text-Conditioned 3D Indoor Scene Synthesis

[Hou In Ivan Tam](https://iv-t.github.io/), [Hou In Derek Pun](https://houip.github.io/), [Austin T. Wang](https://atwang16.github.io/), [Angel X. Chang](https://angelxuanchang.github.io/), [Manolis Savva](https://msavva.github.io/)

<!-- <img src="docs/static/images/teaser.webp" alt="teaser" style="width:100%"/> -->

[Page](https://3dlg-hcvc.github.io/SceneEval/) | [Paper](https://arxiv.org/abs/2503.14756) | [Data](https://github.com/3dlg-hcvc/SceneEval/releases)

## Todo List
- [x] Add documentation for the scene state format
- [x] Provide script for downloading and processing Holodeck's assets
- [ ] Create guide for extending SceneEval with new methods and metrics

## Getting Started

### 1. Environment Setup
First, create and activate the conda environment:
```bash
conda env create -f environment.yaml
conda activate scene_eval
```

### 2. Download Blender
SceneEval requires **Blender 4.2 LTS** for 3D scene rendering and visualization.

Download and install Blender 4.2 LTS from the [official Blender website](https://www.blender.org/download/lts/4-2/).

### 3. Environment Configuration
Create a `.env` file in the root directory following the template in `.env.example`. This file should contain a path to your Blender installation and your OpenAI API key.

### 4. Download Dataset
Download the SceneEval-500 annotations from this repository's [Releases](https://github.com/3dlg-hcvc/SceneEval/releases) page and place the `annotations.csv` file in the `input` directory. Your structure should look like:
```
.
├── input
│   ├── human.glb
│   ├── empty_scene.json
│   ├── annotations.csv
```

**Dataset Composition:**
- **SceneEval-100**: The first 100 entries (IDs 0-99) are manually created
- **SceneEval-500**: The full dataset includes 400 additional entries generated semi-automatically using a VLM

If you wish to use only the first 100 entries, you can edit `annotations.csv` to keep only rows with IDs 0-99.

### 5. Download 3D Assets

SceneEval needs 3D assets to recreate scenes from different generation methods. Download the assets required for the methods you want to evaluate:

**For 3D-FUTURE methods** (ATISS, DiffuScene, LayoutGPT, InstructScene):
1. Visit the [3D-FUTURE dataset page](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future)
2. Follow their download instructions
3. Place the downloaded assets in `_data/3D-FUTURE-model/`

**For Objathor methods** (Holodeck):
Run our automated download script:
```bash
python scripts/prepare_objathor.py

# On Linux, you may see an `directory not empty` error; this is an issue in the original Objathor download script and can be ignored. Simply enter 'y' and press Enter when prompted.
```

Your final `_data` directory structure should look like:
```
.
└── _data
    ├── 3D-FUTURE-model
    │   ├── 0a0f0cf2-3a34-4ba2-b24f-34f361c36b3e
    │   │   ├── raw_model.obj
    │   │   ├── model.mtl
    │   │   ├── texture.png
    │   │   └── ...
    │   ├── ...
    │   └── model_info.json
    │
    └── objathor-assets
        ├── 0a0a8274693445a6b533dce7f97f747c
        │   ├── 0a0a8274693445a6b533dce7f97f747c.glb
        │   ├── ...
        ├── ...
        └── annotations.json
```

## Quick Start Demo

Try SceneEval with our provided example scenes:

### 1. Set Up Input Data
Copy the example scenes to your input directory:
```bash
cp -r input_example/* input/
```

### 2. Configure the Evaluation
Edit `configs/config.yaml` to use the demo evaluation plan:
```yaml
evaluation_plan: eval_plan
```

Then edit `configs/evaluation_plan/eval_plan.yaml` to customize what to evaluate:
- **Methods**: Uncomment the methods you want to test (`ATISS`, `DiffuScene`, `LayoutGPT`, `InstructScene`)
- **Scenes**: For a quick test, set `scene_mode: list` and `scene_list: [0]` to evaluate just the first scene

### 3. Run the Demo
```bash
python main.py
```

Results will be saved to `./output_eval`.

## Evaluating Your Own Scene Generation Method

SceneEval can easily evaluate your own scene generation methods by following these steps:

### 1. Convert Your Scenes to Scene State Format

SceneEval uses the [Scene State format](https://github.com/smartscenes/sstk/wiki/Scene-State-Format) to represent 3D scenes. This JSON format describes:
- Room architecture (walls, doors, windows)
- Object placements

**Quick Start:** Use our [scene state template](./scene_state_template.json) as a starting point. This template shows all required fields and structure.

**Key Requirements:**
- Each scene must be a separate JSON file
- Objects need valid 3D asset IDs that SceneEval can retrieve
- Room architecture must define walls, floors, and any openings

### 2. Organize Your Scene Files

Create a directory structure under `input/` for your method:

```
input/
├── annotations.csv          # SceneEval dataset annotations
├── YOUR_METHOD_NAME/
│   ├── scene_0.json         # Scene for annotation ID 0
│   ├── scene_1.json         # Scene for annotation ID 1
│   └── ...
```

**Important:** Scene filenames must match annotation IDs (e.g., `scene_0.json` for annotation with `id=0`).

### 3. Configure Your Method

Add your method to `configs/models.yaml`:

```yaml
models:
  YOUR_METHOD_NAME:
    input_dir_name: YOUR_METHOD_NAME    # Matches folder name in input/
    output_dir_name: ${input_name}      # Where results will be saved
    asset_datasets:                     # 3D assets your method uses
      - threed_future                   # For 3D-FUTURE assets
      # - objathor                      # For Objathor assets
      # - your_custom_dataset           # For custom assets (see step 4)
```

### 4. (Optional) Add Support for Custom 3D Assets

If your method uses custom 3D assets not covered by 3D-FUTURE or Objathor:

1. Create a new asset dataset class in `assets/` that extends `BaseAssetsDataset`
2. Implement methods to map object IDs to asset file paths and descriptions  
3. See `assets/threed_future.py` as an example implementation

*Need help? Open an issue or submit a pull request to add your dataset!*

### 5. Configure the Evaluation Plan

Edit the `configs/evaluation_plan/eval_plan.yaml` file to set up your evaluation:

```yaml
evaluation_cfg:
  metrics_cfg:
    # Uncomment metrics you want to run
    - ObjCountMetric
    - ObjAttributeMetric
    # - CollisionMetric
    - ...
  input_cfg:
    scene_methods:
      # Uncomment methods you want to evaluate
      # - ATISS
      # - DiffuScene
      # - InstructScene
      # - LayoutGPT
      # - Holodeck
      - YOUR_METHOD_NAME  # Your method name
    
    # Evaluate "all" scenes / "range" for a range / "list" for specific scenes
    scene_mode: "all"
```

Update `configs/config.yaml` to use your plan:
```yaml
evaluation_plan: eval_plan  # Name of your evaluation plan file
```

### 6. Run the Evaluation

Execute the evaluation:
```bash
python main.py
```

Results will be saved to `output_eval/YOUR_METHOD_NAME/` by default.

## Extending SceneEval

SceneEval is built to be extensible! You can easily add new scene generation methods, evaluation metrics, and assets. 

Found a bug or want to contribute a new method or metric? We'd love your help! Please open an issue or submit a pull request. 

We're working on comprehensive documentation for extending SceneEval and will have it available soon.

## Citation
If you find SceneEval helpful in your research, please cite our work:
```
@article{tam2025sceneeval,
    title = {{SceneEval}: Evaluating Semantic Coherence in Text-Conditioned {3D} Indoor Scene Synthesis},
    author = {Tam, Hou In Ivan and Pun, Hou In Derek and Wang, Austin T. and Chang, Angel X. and Savva, Manolis},
    year = {2025},
    eprint = {2503.14756},
    archivePrefix = {arXiv}
}
```

**Note:** When using SceneEval in your work, we encourage you to specify which version of the code and dataset you are using (e.g., commit hash, release tag, or dataset version) to ensure reproducibility and proper attribution.

## Acknowledgements
This work was funded in part by the Sony Research Award Program, a CIFAR AI Chair, a Canada Research Chair, NSERC Discovery Grants, and enabled by support from the [Digital Research Alliance of Canada](https://alliancecan.ca/).
We thank Nao Yamato, Yotaro Shimose, and other members on the Sony team for their feedback.
We also thank Qirui Wu, Xiaohao Sun, and Han-Hung Lee for helpful discussions.
