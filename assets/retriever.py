from .base import DatasetConfig, AssetInfo
from .registry import DatasetRegistry

class Retriever:
    """
    A retriever class that manages asset datasets.
    """

    def __init__(self, dataset_cfgs: dict[str, DatasetConfig]) -> None:
        """
        Initialize the retriever with dataset configurations.
        This class is responsible for managing multiple datasets and retrieving asset information.

        Args:
            dataset_config: the configuration for the dataset
        """
        
        self.dataset_map = {}
        for dataset_name, dataset_cfg in dataset_cfgs.items():
            dataset_class = DatasetRegistry.get_dataset_class(dataset_name)
            self.dataset_map[dataset_cfg.asset_id_prefix] = dataset_class(dataset_cfg)

    def get_asset_info(self, asset_id: str) -> AssetInfo:
        """
        Retrieve information about an asset.

        Args:
            asset_id: the ID of the asset

        Returns:
            AssetInfo object containing the asset's information
        """
        
        asset_id_prefix, asset_id = asset_id.split(".")
        if asset_id_prefix not in self.dataset_map:
            raise ValueError(f"Dataset with prefix '{asset_id_prefix}' not found.")
        
        dataset = self.dataset_map[asset_id_prefix]
        
        return dataset.get_asset_info(asset_id)
