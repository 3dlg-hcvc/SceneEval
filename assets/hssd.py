import csv
import json
import numpy as np
from pathlib import Path
from .base import BaseAssetDataset, DatasetConfig, AssetInfo
from .registry import register_dataset

@register_dataset("hssd")
class HSSDAssetDataset(BaseAssetDataset):
    """
    Dataset for HSSD assets.
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
                reader = csv.DictReader(f)
                self.metadata = {row["id"]: row for row in reader}
        else:
            raise FileNotFoundError(f"Metadata file {self.metadata_path} not found.")
            
    def get_asset_info(self, asset_id: str) -> AssetInfo:
        """
        Get information about the asset.

        Args:
            asset_id: the ID of the asset

        Returns:
            asset_info: an AssetInfo object containing the asset's information
        """
        
        # Determine file path based on whether it's in the group of decomposed assets or not
        if not "part" in asset_id:
            file_path = self.root_dir / "glb" / asset_id[0] / f"{asset_id}.glb"
        else:
            non_part_model_id = asset_id.split("_")[0]
            file_path = self.root_dir / "decomposed" / non_part_model_id / f"{asset_id}.glb"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Asset file {file_path} not found.")
        
        asset_metadata = self.metadata.get(asset_id, {})
        asset_description = asset_metadata.get("wnsynsetkey", "generic object category")
        if asset_metadata.get("name", "") != "":
            asset_description += f" - {asset_metadata['name']}"
        
        # If the model has up and front axes defined, the mesh is not aligned to the canonical axes.
        # We need to create a transformation matrix to align it.
        if asset_metadata["up"] != "" and asset_metadata["front"] != "":
            asset_up_axis = np.array([float(x) for x in asset_metadata["up"].split(",")])
            asset_front_axis = np.array([float(x) for x in asset_metadata["front"].split(",")])
            
            ASSET_CANONICAL_RIGHT = np.array([1, 0, 0]) # +X right
            ASSET_CANONICAL_FRONT = np.array([0, 0, 1]) # +Z front
            ASSET_CANONICAL_UP = np.array([0, 1, 0])    # +Y up
            
            extra_rotation_transform = self._create_alignment_transform(asset_up_axis, asset_front_axis,
                                                                        ASSET_CANONICAL_RIGHT,
                                                                        ASSET_CANONICAL_FRONT,
                                                                        ASSET_CANONICAL_UP)
        else:
            extra_rotation_transform = None
        
        asset_info = AssetInfo(
            asset_id=asset_id,
            file_path=file_path,
            description=asset_description,
            extra_rotation_transform=extra_rotation_transform
        )
        return asset_info

    def _create_alignment_transform(self,
                                    asset_up_axis: np.ndarray,
                                    asset_front_axis: np.ndarray,
                                    asset_canonical_right: np.ndarray,
                                    asset_canonical_front: np.ndarray,
                                    asset_canonical_up: np.ndarray) -> np.ndarray:
        """
        Create a transformation matrix to align an asset's coordinate system to our canonical coordinate system.
        
        This method handles the conversion between two coordinate systems:
        - Asset canonical frame: typically +X right, +Y up, +Z front (varies by asset format)
        - Our canonical frame: +X right, -Y front, +Z up
        
        The transformation is computed in several steps:
        1. Build the asset's local orientation from its up and front vectors (in asset canonical frame)
        2. Create a basis transformation from asset canonical to our canonical
        3. Combine these to get the final transformation matrix
                
        Args:
            asset_up_axis: the asset's up direction vector (relative to asset's canonical frame)
            asset_front_axis: the asset's front direction vector (relative to asset's canonical frame)
            asset_canonical_right: the right vector of the asset's canonical coordinate system
            asset_canonical_front: the front vector of the asset's canonical coordinate system
            asset_canonical_up: the up vector of the asset's canonical coordinate system
            
        Returns:
            transform: 4x4 transformation matrix with a rotation component that transforms from
                       asset's local space to the our canonical coordinate system
        """
        
        # Our canonical axes
        OUR_CANONICAL_RIGHT = np.array([1, 0, 0])   # X-right
        OUR_CANONICAL_FRONT = np.array([0, -1, 0])  # -Y-front
        OUR_CANONCIAL_UP = np.array([0, 0, 1])      # Z-up
        
        # ---------------------------------------------------
        # 1 - Build the asset's local orientation basis in its canonical frame
        
        # Normalize the asset's up and front vectors
        up_normalized = asset_up_axis / np.linalg.norm(asset_up_axis)
        front_normalized = asset_front_axis / np.linalg.norm(asset_front_axis)
        
        # Computer the right vector via cross product: up × front = right
        right_normalized = np.cross(up_normalized, front_normalized)
        right_normalized = right_normalized / np.linalg.norm(right_normalized)
        
        # Recompute front vector to ensure orthogonality: right × up = front
        front_normalized = np.cross(right_normalized, up_normalized)
        front_normalized = front_normalized / np.linalg.norm(front_normalized)
        
        # Orientation matrix: columns are right, front, up in asset's canonical frame
        asset_orientation_in_asset_canonical = np.column_stack([right_normalized, front_normalized, up_normalized])
        
        # ---------------------------------------------------
        # 2 - Build transformation from asset canonical frame to our canonical frame
        
        # Each basis matrix is: columns = [right, front, up]
        asset_canonical_basis = np.column_stack([asset_canonical_right, asset_canonical_front, asset_canonical_up])
        our_canonical_basis = np.column_stack([OUR_CANONICAL_RIGHT, OUR_CANONICAL_FRONT, OUR_CANONCIAL_UP])
        
        # Transform from asset canonical frame to our canonical frame:
        # This describes how to map vectors from asset canonical to our canonical
        asset_canonical_to_app_canonical = our_canonical_basis @ asset_canonical_basis.T
        
        # ---------------------------------------------------
        # 3 - Combine to get final orientation in our canonical frame
        
        # This expresses the asset's orientation (defined in asset canonical frame) in our canonical frame
        final_orientation = asset_canonical_to_app_canonical @ asset_orientation_in_asset_canonical
        
        # Convert orientation matrix into a rotation matrix from asset-local to our-canonical
        # (i.e., R_app_asset = our_basis @ final_orientation.T)
        rotation_matrix = our_canonical_basis @ final_orientation.T
        
        # ---------------------------------------------------
        # 4 - Create the 4x4 transformation matrix (no translation, just rotation)
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        
        return transform
