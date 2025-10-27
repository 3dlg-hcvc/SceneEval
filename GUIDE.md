# Guide for Extending SceneEval
This guide provides a step-by-step example of how to extend SceneEval to support a new method and its assets.

We will show how we added *LayoutVLM* to SceneEval as an example.

## Adding a New Asset Dataset

LayoutVLM uses a custom set of 3D assets.
To enable SceneEval to evaluate LayoutVLM scenes, we first need to add support for retrieving these assets.

### 1. Add an Entry in `configs/assets.yaml`
`configs/assets.yaml` defines metadata for all asset datasets SceneEval can use.
To add LayoutVLM's assets, we add a new entry:
```yaml
assets:
  layoutvlm-objathor:                               # Name of the asset dataset
    asset_id_prefix: "layoutvlm-objaverse"          # Prefix for asset IDs used in scene state files
    dataset_root_path: "_data/layoutvlm-objathor"   # Path to the dataset root directory
    dataset_metadata_path: null                     # (Optional) Path to a central metadata file if available
```

<details>
<summary><strong>
Explanations
</strong></summary>

- `layoutvlm-objathor`: This is the internal name we give to this asset dataset. We will reference this name later when configuring methods that use these assets.
- `asset_id_prefix`: This prefix is used in scene state files to identify assets from this dataset. For example, an object with ID `layoutvlm-objaverse.123456` indicates it comes from this dataset and has the unique ID of `123456`.
- `dataset_root_path`: This is the local path where the asset files are stored. SceneEval will look here to find the 3D models when reconstructing scenes.
- `dataset_metadata_path`: If the dataset has a central metadata file (e.g., a CSV or JSON file listing all assets and their properties), you can specify its path here. If not, set it to `null`. In this case, LayoutVLM does not have a central metadata file.

</details>



### 2. Add an Asset Dataset Class under `assets/`
An asset dataset class defines how to get the file paths and descriptions for objects in this dataset.
To add support for LayoutVLM's assets, we create a new class `LayoutVLMObjathorAssetsDataset` in `assets/layoutvlm_objathor.py`, extending `BaseAssetsDataset`:

```python
import json
from pathlib import Path
from .base import BaseAssetDataset, DatasetConfig, AssetInfo
from .registry import register_dataset

@register_dataset("layoutvlm_objathor")
class LayoutVLMObjathorAssetDataset(BaseAssetDataset):
    """
    Dataset for LayoutVLM-Objathor assets.
    """

    def __init__(self, dataset_config: DatasetConfig) -> None:
        """
        Initialize the asset dataset.

        Args:
            dataset_config: the configuration for the dataset
        """
        
        self.asset_id_prefix = dataset_config.asset_id_prefix
        self.root_dir = Path(dataset_config.dataset_root_path).expanduser().resolve()
    
    def get_asset_info(self, asset_id: str) -> AssetInfo:
        """
        Get information about the asset.

        Args:
            asset_id: the ID of the asset

        Returns:
            AssetInfo object containing the asset's information
        """
        
        file_path = self.root_dir / asset_id / f"{asset_id}.glb"
        asset_data_json_path = self.root_dir / asset_id / "data.json"
        
        if asset_data_json_path.exists() is False:
            raise FileNotFoundError(f"Asset data file {asset_data_json_path} not found.")

        with open(asset_data_json_path, "r") as f:
            asset_data_json = json.load(f)
        metadata = asset_data_json["annotations"]

        asset_description = f"{metadata['category']}, {metadata['description']}, {metadata['materials']}"
        
        return AssetInfo(
            asset_id=asset_id,
            file_path=file_path,
            description=asset_description,
            extra_rotation_transform=None
        )
```

<details>
<summary><strong>
Explanations
</strong></summary>

SceneEval uses the `BaseAssetDataset` class as a base for all asset datasets.

The `@register_dataset("layoutvlm_objathor")` decorator registers this class under the name `layoutvlm_objathor`, which matches the key we used in `configs/assets.yaml`. This allows SceneEval to locate this class when it needs to retrieve assets from this dataset.

The `__init__` method initializes the dataset with its configuration, a `DatasetConfig` object that contains the fields we defined in `configs/assets.yaml`.

The required method to implement is `get_asset_info`, which takes an `asset_id` and returns an `AssetInfo` object containing:
- `asset_id`: The unique ID of the asset
- `file_path`: The path to the 3D model file
- `description`: A textual description of the asset (from metadata)
- `extra_rotation_transform`: Any additional rotation needed to align the model when loading (set to `None` if not needed, as in this case)

The description should provide useful information that can help VLMs to understand the object.
LayoutVLM does not have a central metadata file, instead each asset has a `data.json` file containing its metadata.
We extract the category, description, and materials fields from this file to form the asset description.

If there are any special preprocessing steps needed, for example computing the extra rotation transform based on the asset's metadata, it should be done here (See `assets/hssd.py` for an example).

</details>



### 3. Add the Dataset Class to `assets/__init__.py`
To make the new dataset class discoverable, we need to import it in `assets/__init__.py`.

Add `layoutvlm_objathor` to the imports:

```python
from .retriever import Retriever

# Import all dataset implementations to ensure they are registered
from . import threed_future, objathor, layoutvlm_objathor, hssd

__all__ = [
    "Retriever",
]
```



## Adding a New Method
To make SceneEval support results from a new scene generation method, we need to configure the input and output directories, as well as the asset datasets it uses.

### Add an Entry in `configs/models.yaml`
`configs/models.yaml` defines all scene generation methods SceneEval can evaluate.
To add LayoutVLM, we add a new entry:
```yaml
models:
  LayoutVLM:                                                # Name of the method
    input_dir_name: LayoutVLM                               # Matches folder name in input/
    output_dir_name: ${models.LayoutVLM.output_dir_name}    # Where results will be saved under output_eval/
    asset_datasets:                                         # assets the method uses (keys in configs/assets.yaml)
      - layoutvlm_objathor
```

<details>
<summary><strong>
Explanations
</strong></summary>

- `LayoutVLM`: This is the internal name we give to this method. We will reference this name later when configuring the evaluation plan.
- `input_dir_name`: This specifies the folder name under `input/` where you will place the scene files generated by this method.
- `output_dir_name`: This specifies where the evaluation results for this method will be saved under `output_eval/`. Here, we use a variable reference to set it to the same as `input_dir_name`.
- `asset_datasets`: This lists the asset datasets that this method uses. Valid entries are the keys defined in `configs/assets.yaml`. Here, we specify `layoutvlm_objathor` since LayoutVLM uses this dataset.

</details>



## Preparing Your Method's Output for Evaluation

SceneEval uses the [Scene State format](https://github.com/smartscenes/sstk/wiki/Scene-State-Format) to represent 3D scenes. This JSON format describes:
- Room architecture (walls, doors, windows)
- Object placements

### Quick Start
We provide a [scene state template](./scene_state_template.json) to help you get started.
It shows all required fields and the structure of a scene state file.

### Requirements for Scene State Files
When preparing your method's output for evaluation, ensure the following:
- Each output scene must be a separate JSON file, with filenames matching the format `scene_{ID}.json`
    - `{ID}` corresponds to the annotation entry ID in `annotations.csv`.
- Objects listed in the scene state file must have prefixed assets IDs that match those defined in `configs/assets.yaml`
    - For LayoutVLM, this means using the `layoutvlm-objaverse` prefix.
    - For example, an object with ID `layoutvlm-objaverse.88be8ad75f274c7b8da4a54fef3aaac9` indicates SceneEval should use the `layoutvlm_objathor` dataset class we defined earlier to retrieve the asset `88be8ad75f274c7b8da4a54fef3aaac9`.

## Running Evaluation
You are now ready to run SceneEval on your method's output!


### 1. Organize Your Scene Files

Place your methods' scene state JSON files under the appropriate input directory, as defined in `configs/models.yaml`.
For LayoutVLM, the directory structure should look like this:

```
input
├── annotations.csv          # SceneEval dataset annotations
├── LayoutVLM/
│   ├── scene_0.json         # Scene for annotation ID 0
│   ├── scene_1.json         # Scene for annotation ID 1
│   └── ...
...
```

### 2. Configure a Evaluation Plan

Edit or create a new evaluation plan file under `configs/evaluation_plan/` to add your method and specify which metrics to run.

For example, we can edit `no_llm_plan.yaml` to include LayoutVLM, by adding it to the `scene_methods` list:

```yaml
evaluation_cfg:
  metrics:
    # - ObjCountMetric
    # - ObjAttributeMetric
    # - ObjObjRelationshipMetric
    # - ObjArchRelationshipMetric
    - CollisionMetric
    # - SupportMetric
    - NavigabilityMetric
    # - AccessibilityMetric
    - OutOfBoundMetric
    - OpeningClearanceMetric
  output_dir: ./output_eval
  save_blend_file: True
  vlm: GPT
  use_existing_matching: True
  use_empty_matching_result: True
  support_metric_use_existing_support_type_assessment: False
  no_eval: False
  verbose: False

input_cfg:
  root_dir: ./input
  scene_methods:
    # - ATISS
    # - DiffuScene
    # - InstructScene
    # - LayoutGPT
    # - Holodeck
    # - HSM
    - LayoutVLM  # <------------------------------------ Add LayoutVLM here
  method_use_simple_architecture: []
  scene_mode: range # all, range, list
  scene_range: [0, 5] # Left inclusive, right exclusive
  scene_list: []
  annotation_file: ./input/annotations.csv

render_cfg:
  normal_render_tasks:
    - scene_top
    # - obj_solo
    # - obj_size
    # - obj_surroundings
    # - obj_global_top
  semantic_render_tasks:
    # - scene_top
    # - obj_solo
    # - obj_size
    # - obj_surroundings
    # - obj_global_top
  semantic_color_reference: null
```



### 3. Run the Evaluation

You are now ready to run the evaluation!
There are two ways to specify the evaluation plan to use when running `main.py`.

#### Option 1) Via command line argument using `Hydra` syntax:
```bash
python main.py evaluation_plan=no_llm_plan
```

#### Option 2) By setting the default evaluation plan in `configs/config.yaml`:
```yaml
defaults:
  - evaluation_plan: no_llm_plan
```
Then, execute the command without needing to specify the plan:
```bash
python main.py
```

Evaluation results will be saved to `output_eval/LayoutVLM/`, as defined in `configs/models.yaml`.
