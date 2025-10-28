# SceneEval

### SceneEval: Evaluating Semantic Coherence in Text-Conditioned 3D Indoor Scene Synthesis

[Hou In Ivan Tam](https://iv-t.github.io/), [Hou In Derek Pun](https://houip.github.io/), [Austin T. Wang](https://atwang16.github.io/), [Angel X. Chang](https://angelxuanchang.github.io/), [Manolis Savva](https://msavva.github.io/)

<!-- <img src="docs/static/images/teaser.webp" alt="teaser" style="width:100%"/> -->

[Page](https://3dlg-hcvc.github.io/SceneEval/) | [Paper](https://arxiv.org/abs/2503.14756) | [Data](https://github.com/3dlg-hcvc/SceneEval/releases)



## News
- 2025-10-27: Release v1.1 with a new metric *Opening Clearance*, support for [LayoutVLM](https://github.com/sunfanyunn/LayoutVLM) and [HSM](https://github.com/3dlg-hcvc/hsm), bug fixes, and more! The environment setup is now simplified and the demo is easier to run! Give it a try!
- 2025-06-27: Codebase release v1.0!
- 2025-06-10: Released SceneEval-500 dataset and v0.9 of the SceneEval codebase!



## Todo List
- [x] Add documentation for the scene state format
- [x] Provide script for downloading and processing Holodeck's assets
- [x] Create guide for extending SceneEval with new methods and metrics
- [ ] Replace custom VLM interface with Pydantic AI



## Environment Setup

### 1. Environment Setup
First, create and activate the [conda](https://www.anaconda.com/docs/getting-started/miniconda/main) environment:
```bash
conda env create -f environment.yaml
conda activate scene_eval
```

### 2. (Optional) Setup OpenAI API Key
SceneEval requires an VLM to run certain metrics.

Metrics that DO NOT require a VLM are:
- *Collision*, *Navigability*, *Out of Bounds*, and *Opening Clearance* 

Metrics that REQUIRE a VLM are:
- *Object Count*, *Object Attribute*, *Object-Object Relationship*, *Object-Architecture Relationship*, *Object Support*, *Object Accessibility*

To run the metrics that require a VLM, the default implementation uses OpenAI's GPT-4o, so you will need an OpenAI API key. (The demo below does not require this.)

Create a `.env` file in the root directory following the template in `.env.example`, and add your OpenAI API key:
```
OPENAI_API_KEY=<your_openai_api_key_here>
```


## Dataset and 3D Assets Setup

### 1. Download SceneEval-500 Dataset
Download the SceneEval-500 annotations from this repository's [Releases](https://github.com/3dlg-hcvc/SceneEval/releases/tag/SceneEval-500_v250610) page and place the `annotations.csv` file in the `input` directory. The structure should look like:
```
.
└── input
    ├── annotations.csv
    ├── empty_scene.json
    ├── hotel_room_1k.exr
    ├── human.glb
    └── ...
```

**Dataset Composition:**
- **SceneEval-100**: The first 100 entries (IDs 0-99) are manually created
- **SceneEval-500**: The full dataset includes 400 additional entries generated semi-automatically using a VLM

### 2. Download 3D Assets

SceneEval needs 3D assets to recreate scenes from different generation methods.
You only need to download the assets for the methods you want to evaluate.

<details>
<summary><strong>
For 3D-FUTURE methods
<a href="https://github.com/nv-tlabs/ATISS">ATISS</a>,
<a href="https://github.com/tangjiapeng/DiffuScene">DiffuScene</a>,
<a href="https://github.com/weixi-feng/LayoutGPT">LayoutGPT</a>,
<a href="https://github.com/chenguolin/InstructScene">InstructScene</a>
</strong></summary>

1. Visit the [3D-FUTURE dataset page](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future)
2. Follow their download instructions
3. Place the downloaded assets in `_data/3D-FUTURE-model/`

</details>

<details>
<summary><strong>
For <a href="https://github.com/allenai/Holodeck">Holodeck</a>
</strong></summary>

Run our automated download script to download and preprocess the Objathor assets they use:
```bash
python scripts/prepare_objathor.py

# On Linux, you may see an `directory not empty` error; this is an issue in the original Objathor download script and can be ignored. Simply enter 'y' and press Enter when prompted.
```
</details>

<details id="layoutvlm">
<summary><strong>
For <a href="https://github.com/sunfanyunn/LayoutVLM">LayoutVLM</a>
</strong></summary>

1. Download their preprocessed assets from [their repo](https://github.com/sunfanyunn/LayoutVLM?tab=readme-ov-file#data-preprocessing).
2. Unzip and place the contents in `_data/layoutvlm-objathor/`

</details>


<details>
<summary><strong>
For <a href="https://github.com/3dlg-hcvc/HSM">HSM</a>
</strong></summary>

HSM uses assets from the [Habitat Synthetic Scenes Dataset (HSSD)](https://3dlg-hcvc.github.io/hssd/).
Some assets are compressed with *KHR_texture_basisu*, which is currently not supported by Blender.
We provide a script to download and decompress the assets into Blender-compatible GLB files.

Prerequisites:
1. [Agree to the HSSD dataset license on HuggingFace](https://huggingface.co/datasets/hssd/hssd-models)
       
2. Set up to clone repos from HuggingFace using SSH or HTTPS:
    - For SSH: [set up SSH keys on your machine and add the public key to your HuggingFace account](https://huggingface.co/docs/hub/en/security-git-ssh)
    - For HTTPS: [prepare to enter your HuggingFace access token with write permissions when prompted](https://huggingface.co/docs/hub/en/security-tokens)

3. Install *gltf-transform* and *ktx* command line tools and ensure they are in your PATH:
    - [gltf-transform](https://www.npmjs.com/package/@gltf-transform/cli) via npm: `npm install -g @gltf-transform/cli`
    - [ktx](https://github.com/KhronosGroup/KTX-Software/releases) from the KhronosGroup/KTX-Software repository

Then run the script:
```bash
python scripts/prepare_hsm.py
```

</details>

---
<details>
<summary><strong>
Your <code>_data/</code> directory should look like this if you have all assets downloaded
</strong></summary>

```
_data
├── 3D-FUTURE-model
│   ├── model_info.json
│   ├── 0a0f0cf2-3a34-4ba2-b24f-34f361c36b3e
│   |   ├── raw_model.obj
│   |   ├── model.mtl
│   |   ├── texture.png
│   |   ├── ...
│   ├── ...
├── objathor-assets
│   ├── annotations.json
│   ├── 0a0a8274693445a6b533dce7f97f747c
│   |   ├── 0a0a8274693445a6b533dce7f97f747c.glb
│   |   ├── ...
├── layoutvlm-objathor
│   ├── 0a3dc72fb1bb41439005cac4a3ebb765
│   |   ├── 0a3dc72fb1bb41439005cac4a3ebb765.glb
│   |   ├── data.json
│   |   ├── ...
│   ├── ...
└── hssd
    ├── fpmodels.csv
    ├── glb
     │   ├── 0
     │   |   ├── 0a0b9f607345b6cee099d681f28e19e1f0a215c8.glb
     │   |   ├── ...
     │   ├── ...
     └── decomposed
         ├── 00a2b0f3886ccb5ffddac704f8eeec324a5e14c6
         |   ├── 00a2b0f3886ccb5ffddac704f8eeec324a5e14c6_part_1.glb
         |   ├── 00a2b0f3886ccb5ffddac704f8eeec324a5e14c6_part_2.glb
         |   ├── ...
         ├── ...
```

</details>


## Quick Start Demo

Try SceneEval with our provided example scenes. You do *not* need an OpenAI API key for this demo.

### 1. Download the LayoutVLM Assets
Follow the instructions in the [*For LayoutVLM*](#layoutvlm) section above to download the LayoutVLM assets.

### 2. Run the Demo
```bash
# Copy provided example scenes to input directory
cp -r input_example/* input/

# Run SceneEval
python main.py
```
This will run the evaluation on five example scenes generated by LayoutVLM using the `no_llm_plan` evaluation plan, which runs the following metrics: *Collision*, *Navigability*, *Out of Bounds*, and *Opening Clearance*.

Results will be saved to `./output_eval`.



## Extending SceneEval to a New Method or Dataset

SceneEval is built to be extensible! You can easily add new scene generation methods, evaluation metrics, and assets.

**[Follow this step-by-step guide to see how to add a new method that uses a new 3D asset source.](./GUIDE.md)**



## Contributing to SceneEval

Found a bug or want to contribute a new method or metric? We'd love your help! Please open an issue or submit a pull request. 



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
