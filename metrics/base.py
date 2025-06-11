from abc import ABC, abstractmethod
from dataclasses import dataclass
from pydantic import BaseModel
from typing import Any

@dataclass
class MetricResult:
    """
    The result of running a metric.

    Attributes:
        message: the message of the result
        data: the data of the result
    """

    message: str = ""
    data: dict[str, any] | BaseModel = None

class BaseMetric(ABC):
    """
    Base class for metrics.
    """

    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the metric.

        Args:
            **kwargs: the arguments to initialize the metric
        """

        raise NotImplementedError

    # @abstractmethod
    def run(self, verbose: bool = False) -> MetricResult:
        """
        Run the metric.

        Args:
            verbose: whether to visualize during the run

        Returns:
            result: the result of running the metric
        """
        
        raise NotImplementedError
