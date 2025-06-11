import json
from pathlib import Path
from .base import BaseAssetDataset, DatasetConfig, AssetInfo
from .registry import register_dataset

@register_dataset("objathor")
class ThreeDFutureAssetDataset(BaseAssetDataset):
    """
    Dataset for 3D-Future assets.
    """

    def __init__(self, dataset_config: DatasetConfig) -> None:
        """
        Initialize the asset dataset.

        Args:
            dataset_config: the configuration for the dataset
        """
        
        self.asset_id_prefix = dataset_config.asset_id_prefix
        self.root_dir = Path(dataset_config.dataset_root_path).expanduser().resolve()
        self.metadata_path = Path(dataset_config.dataset_metadata_path).expanduser().resolve()
        
        if self.metadata_path.exists():
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)
        else:
            raise FileNotFoundError(f"Metadata file {self.metadata_path} not found.")
    
    def get_asset_info(self, asset_id: str) -> AssetInfo:
        """
        Get information about the asset.

        Args:
            asset_id: the ID of the asset

        Returns:
            AssetInfo object containing the asset's information
        """
        
        file_path = self.root_dir / asset_id / f"{asset_id}.glb"
        
        metadata = self.metadata[asset_id]
        asset_description = metadata["category"]
        if metadata["ref_category"] is not None or metadata["ref_category"] != "":
            asset_description += f" - {metadata['ref_category']}"
        if metadata["description"] is not None or metadata["description"] != "":
            asset_description += f", {metadata['description']}"
        elif metadata["description_auto"] is not None or metadata["description_auto"] != "":
            asset_description += f", {metadata['description_auto']}"
        
        return AssetInfo(
            asset_id=asset_id,
            file_path=file_path,
            description=asset_description,
            extra_rotation_transform=None
        )
