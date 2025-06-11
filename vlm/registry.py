from .base import BaseVLM
from omegaconf import DictConfig

class VLMRegistry:
    """
    Registry for VLM classes and their configurations.
    """
    
    _vlms: dict[str, type[BaseVLM]] = {}
    _config_classes: dict[str, type] = {}
    
    @classmethod
    def register(cls, vlm_class: type[BaseVLM]) -> None:
        """
        Register an VLM class.
        
        Args:
            vlm_class: the VLM class to register.
        """
        
        vlm_name = vlm_class.__name__
        cls._vlms[vlm_name] = vlm_class
    
    @classmethod
    def register_config(cls, vlm_name: str, config_class: type) -> None:
        """
        Register a configuration class for an VLM.
        
        Args:
            vlm_name: the name of the VLM.
            config_class: the configuration class for the VLM.
        """
        
        cls._config_classes[vlm_name] = config_class
    
    @classmethod
    def create_config(cls, vlm_name: str, config_dict: dict) -> type:
        """
        Create a configuration instance for an VLM.
        
        Args:
            vlm_name: the name of the VLM.
            config_dict: the configuration dictionary from the config file.
            
        Returns:
            An instance of the VLM's configuration class, or None if no config is needed.
        """
        
        if vlm_name in cls._config_classes:
            return cls._config_classes[vlm_name](**config_dict)
        return None
    
    @classmethod
    def get_vlm_class(cls, name: str) -> type[BaseVLM]:
        """
        Get the class of an VLM by its name.
        
        Args:
            name: the name of the VLM.
        
        Returns:
            The class of the VLM.
        
        Raises:
            KeyError: If the VLM is not registered.
        """
        
        if name not in cls._vlms:
            raise KeyError(f"Unknown VLM: {name}. Available VLMs: {list(cls._vlms.keys())}")
        
        return cls._vlms[name]
    
    @classmethod
    def instantiate_vlm(cls, vlm_name: str, vlm_config: DictConfig | None = None, **kwargs) -> BaseVLM:
        """
        Instantiate an VLM with its configuration.
        
        Args:
            vlm_name: the name of the VLM to instantiate.
            vlm_config: the configuration for the VLM, if available.
            **kwargs: additional arguments passed to the VLM constructor.
            
        Returns:
            An instance of the VLM.
        """
        
        vlm_class = cls.get_vlm_class(vlm_name)
        
        # Prepare kwargs for VLM instantiation
        init_kwargs = kwargs.copy()
        
        # Add VLM-specific config if it exists
        if vlm_config is not None and vlm_name in cls._config_classes:
            config_instance = cls.create_config(vlm_name, vlm_config)
            if config_instance is not None:
                init_kwargs["cfg"] = config_instance
        elif vlm_name in cls._config_classes:
            # If no config is provided but a config class exists, create a default config
            init_kwargs["cfg"] = cls.create_config(vlm_name, {})
        
        return vlm_class(**init_kwargs)
    
    @classmethod
    def get_available_vlms(cls) -> list[str]:
        """
        Get a list of all registered VLM names.
        
        Returns:
            A list of registered VLM names.
        """
        
        return list(cls._vlms.keys())

# Decorator
def register_vlm(config_class: type | None = None):
    """
    Decorator to register an VLM class.
    
    Args:
        config_class: optional configuration class for the VLM.
    """
    
    def decorator(vlm_class: type[BaseVLM]):
        # Register the VLM
        VLMRegistry.register(vlm_class)
        
        # Register the config class if provided
        if config_class is not None:
            VLMRegistry.register_config(vlm_class.__name__, config_class)
        
        return vlm_class
    
    return decorator
