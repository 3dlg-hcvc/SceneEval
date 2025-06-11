# SceneEval

### SceneEval: Evaluating Semantic Coherence in Text-Conditioned 3D Indoor Scene Synthesis

[Hou In Ivan Tam](https://iv-t.github.io/), [Hou In Derek Pun](https://houip.github.io/), [Austin T. Wang](https://atwang16.github.io/), [Angel X. Chang](https://angelxuanchang.github.io/), [Manolis Savva](https://msavva.github.io/)

<!-- <img src="docs/static/images/teaser.webp" alt="teaser" style="width:100%"/> -->

[Page](https://3dlg-hcvc.github.io/SceneEval/) | [Paper](https://arxiv.org/abs/2503.14756) | [Data](https://github.com/3dlg-hcvc/SceneEval/releases)

## Todo List
- [ ] Add documentation for the scene state format
- [ ] Provide script for downloading and processing Holodeck's assets
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
Create a `.env` file in the root directory following the template provided in `.env.example`.

### 4. Download Dataset
Download the SceneEval-500 annotations from this repository's [Releases](https://github.com/3dlg-hcvc/SceneEval/releases) page and place the `annotations.csv` file in the `input` directory. Your structure should look like:
```
.
├── input
│   ├── human.glb
│   ├── empty_scene.json
│   ├── annotations.csv
```

### 5. Download 3D Assets
You'll need to download the 3D assets used by various scene generation methods so SceneEval can recreate the scenes.

**3D-FUTURE** ([download here](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future))
- Required for [ATISS](https://arxiv.org/abs/2110.03675), [DiffuScene](https://arxiv.org/abs/2303.14207), [LayoutGPT](https://arxiv.org/abs/2305.15393), and [InstructScene](https://arxiv.org/abs/2402.04717)

Place the downloaded assets in the `_data` directory:
```
.
├── _data
│   ├── 3D-FUTURE-model
│   │   ├── 0a0f0cf2-3a34-4ba2-b24f-34f361c36b3e
│   │   │   ├── raw_model.obj
│   │   │   ├── model.mtl
│   │   │   ├── texture.png
│   │   │   └── ...
│   │   ├── ...
│   │   └── model_info.json
```

## Quick Start

### 1. Set Up Input Data
Copy the contents from `input_example` to the `input` directory.

### 2. Configure Your Evaluation
- Open `configs/config.yaml` and set `evaluation_plan` to `eval_plan`
- Edit `configs/evaluation_plan/eval_plan.yaml` to customize your evaluation:
    - Uncomment the methods you want to evaluate (`ATISS`, `DiffuScene`, `LayoutGPT`, `InstructScene`) under `input_cfg.scene_methods`
    - Set `input_cfg.scene_mode` to `list` and `input_cfg.scene_list` to `[0]` to evaluate just the first scene

### 3. Run the Evaluation
Execute the evaluation script (results will be saved to `output_eval` by default):
```bash
python main.py
```

## Running Your Own Evaluations

### 1. Organize Your Input Data

Place your scene state files in the `input` directory, organized by method. Name each file as `scene_<index>.json`, where `<index>` matches the corresponding annotation entry.

Your directory structure should look like:

```
.
├── input
│   ├── human.glb
│   ├── empty_scene.json
│   ├── METHOD_1
│   │   ├── scene_0.json
│   │   ├── scene_1.json
│   │   └── ...
│   ├── METHOD_2
│   │   ├── scene_0.json
│   │   ├── scene_1.json
│   │   └── ...
```

### 2. Create Your Evaluation Plan
Create or modify a `.yaml` file in `configs/evaluation_plan/` to specify:
- Which methods to evaluate
- Which metrics to run
- Any additional evaluation parameters

Check out the existing files in that directory for examples, then update the `evaluation_plan` field in `configs/config.yaml` to use your plan.

### 3. Run the Evaluation
Execute the evaluation script:
```bash
python main.py
```

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
