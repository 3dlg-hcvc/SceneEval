import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class DatasetConfig:
    """
    Configuration for a dataset.

    Attributes:
        asset_id_prefix: the prefix used to identify assets in this dataset
        dataset_root_path: the root path of the dataset
        dataset_metadata_path: the path to the dataset metadata file
    """

    asset_id_prefix: str
    dataset_root_path: str
    dataset_metadata_path: str

@dataclass
class AssetInfo:
    """
    The information of an asset.

    Attributes:
        asset_id: the ID of the asset
        file_path: the path to the asset file
        description: a description of the asset
        extra_rotation_transform: an optional rotation transform matrix applied to the asset
    """

    asset_id: str
    file_path: Path
    description: str
    extra_rotation_transform: np.ndarray | None = None

class BaseAssetDataset(ABC):
    """
    Base class for an asset dataset.
    """

    @abstractmethod
    def __init__(self, dataset_config: DatasetConfig) -> None:
        """
        Initialize the asset dataset.

        Args:
            dataset_config: the configuration for the dataset
        """
        
        raise NotImplementedError
    
    @abstractmethod
    def get_asset_info(self, asset_id: str) -> AssetInfo:
        """
        Get information about the asset.

        Args:
            asset_id: the ID of the asset

        Returns:
            asset_info: an AssetInfo object containing the asset's information
        """
        
        raise NotImplementedError
