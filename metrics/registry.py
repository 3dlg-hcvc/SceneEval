from .base import BaseMetric
from omegaconf import DictConfig

class MetricRegistry:
    """
    Registry for metrics classes and their configurations.
    """
    
    _metrics: dict[str, type[BaseMetric]] = {}
    _vlm_required: set[str] = set()
    _config_classes: dict[str, type] = {}
    
    @classmethod
    def register(cls, metric_class: type[BaseMetric], requires_vlm: bool) -> None:
        """
        Register a metric class.
        
        Args:
            metric_class: the metric class to register.
            requires_vlm: whether the metric requires a VLM.
        """
        
        metric_name = metric_class.__name__
        cls._metrics[metric_name] = metric_class
        
        if requires_vlm:
            cls._vlm_required.add(metric_name)
    
    @classmethod
    def register_config(cls, metric_name: str, config_class: type) -> None:
        """
        Register a configuration class for a metric.
        
        Args:
            metric_name: the name of the metric.
            config_class: the configuration class for the metric.
        """
        
        cls._config_classes[metric_name] = config_class
    
    @classmethod
    def create_config(cls, metric_name: str, config_dict: dict) -> type[BaseMetric] | None:
        """
        Create a configuration instance for a metric.
        
        Args:
            metric_name: the name of the metric.
            config_dict: the configuration dictionary from the config file.
            
        Returns:
            An instance of the metric's configuration class, or None if no config is needed.
        """
        
        if metric_name in cls._config_classes:
            return cls._config_classes[metric_name](**config_dict)
        return None
    
    @classmethod
    def load_all_configs(cls, metrics_config: DictConfig, metric_names: list[str]) -> dict[str, type]:
        """
        Load all configurations for the specified metrics.
        
        Args:
            metrics_config: the metrics section from the config file.
            metric_names: list of metric names to load configs for.
            
        Returns:
            A dictionary mapping metric names to their configuration instances.
        """
        
        configs = {}
        for metric_name in metric_names:
            if hasattr(metrics_config, metric_name):
                config_dict = getattr(metrics_config, metric_name)
                config_instance = cls.create_config(metric_name, config_dict)
                if config_instance is not None:
                    configs[metric_name] = config_instance
        
        return configs
    
    @classmethod
    def get_metric_class(cls, name: str) -> type[BaseMetric]:
        """
        Get the class of a metric by its name.
        
        Args:
            name: the name of the metric.
        
        Returns:
            The class of the metric.
        
        Raises:
            KeyError: If the metric is not registered.
        """
        
        if name not in cls._metrics:
            raise KeyError(f"Unknown metric: {name}. Available metrics: {list(cls._metrics.keys())}")
        
        return cls._metrics[name]
    
    @classmethod
    def instantiate_metric(cls, metric_name: str, metric_configs: dict[str, type], **common_kwargs) -> BaseMetric:
        """
        Instantiate a metric with its configuration and common arguments.
        
        Args:
            metric_name: the name of the metric to instantiate.
            metric_configs: dictionary of all metric configurations.
            **common_kwargs: common arguments passed to all metrics (scene, vlm, etc.).
            
        Returns:
            An instance of the metric.
        """
        
        metric_class = cls.get_metric_class(metric_name)
        
        # Prepare kwargs for metric instantiation
        kwargs = common_kwargs.copy()
        
        # Add metric-specific config if it exists
        if metric_name in metric_configs:
            # Use standardized 'cfg' parameter name
            kwargs["cfg"] = metric_configs[metric_name]
        
        return metric_class(**kwargs)
    
    @classmethod
    def get_vlm_metrics(cls) -> dict[str, type[BaseMetric]]:
        """
        Get all metrics that require a VLM.
        
        Returns:
            A dictionary mapping metric names to their classes for VLM-required metrics.
        """
        
        return {name: metric_cls for name, metric_cls in cls._metrics.items() if name in cls._vlm_required}
    
    @classmethod
    def get_non_vlm_metrics(cls) -> dict[str, type[BaseMetric]]:
        """
        Get all metrics that do not require a VLM.
        
        Returns:
            A dictionary mapping metric names to their classes for non-VLM metrics.
        """
        
        return {name: metric_cls for name, metric_cls in cls._metrics.items() if name not in cls._vlm_required}
    
    @classmethod
    def requires_vlm(cls, name: str) -> bool:
        """
        Check if a metric requires a VLM.
        Args:
            name: the name of the metric to check.
        Returns:
            True if the metric requires a VLM, False otherwise.
        """
        
        return name in cls._vlm_required
    
# Decorators
def register_non_vlm_metric(config_class: type | None = None):
    """
    Shortcut decorator to register a metric that does not require a VLM.
    
    Args:
        config_class: optional configuration class for the metric.
    """
    
    def decorator(metric_class: type[BaseMetric]):
        # Register the metric
        MetricRegistry.register(metric_class, requires_vlm=False)
        
        # Register the config class if provided
        if config_class is not None:
            MetricRegistry.register_config(metric_class.__name__, config_class)
        
        return metric_class
    
    return decorator

def register_vlm_metric(config_class: type | None = None):
    """
    Shortcut decorator to register a metric that requires a VLM.
    
    Args:
        config_class: optional configuration class for the metric.
    """
    def decorator(metric_class: type[BaseMetric]):
        # Register the metric
        MetricRegistry.register(metric_class, requires_vlm=True)
        
        # Register the config class if provided
        if config_class is not None:
            MetricRegistry.register_config(metric_class.__name__, config_class)
        
        return metric_class
    
    return decorator
