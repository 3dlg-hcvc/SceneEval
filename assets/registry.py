from .base import BaseAssetDataset

class DatasetRegistry:
    """
    Registry for dataset classes.
    """
    
    _datasets: dict[str, type[BaseAssetDataset]] = {}
    
    @classmethod
    def register(cls, name: str, dataset_class: type[BaseAssetDataset]):
        """
        Register a dataset class with an name.
        
        Args:
            name: the name for the dataset
            dataset_class: the dataset class to register
        """
        
        cls._datasets[name] = dataset_class
    
    @classmethod
    def get_dataset_class(cls, name: str) -> type[BaseAssetDataset]:
        """
        Get the dataset class based on the name.
        
        Args:
            name: the name for the dataset
            
        Returns:
            The dataset class corresponding to the name
        """
        
        if name not in cls._datasets:
            raise ValueError(f"Unknown dataset name: {name}. Available names: {list(cls._datasets.keys())}")
        
        return cls._datasets[name]

# Decorator to register dataset classes
def register_dataset(name: str):
    """
    Decorator to register a dataset class with an name.
    """
    
    def decorator(dataset_class: type[BaseAssetDataset]):
        DatasetRegistry.register(name, dataset_class)
        return dataset_class
    
    return decorator
