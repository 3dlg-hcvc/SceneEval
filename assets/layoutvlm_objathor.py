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
