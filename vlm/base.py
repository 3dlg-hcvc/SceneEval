from abc import ABC, abstractmethod
from pathlib import Path
from pydantic import BaseModel
from typing import Any

class BaseVLM(ABC):
    """
    Base class for VLM interface implementations.
    """

    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the VLM interface.

        Args:
            **kwargs: the arguments to initialize the VLM interface
        """
        
        raise NotImplementedError

    @abstractmethod
    def send(self,
             task: str,
             prompt_info: dict[str, str] | None = None,
             image_paths: list[Path] = [],
             response_format: BaseModel | None = None,
             prepend_string: str | None = None) -> BaseModel | str:
        """
        Send a message to the VLM and return the response.

        Args:
            task: the task to perform
            prompt_info: the information to substitute into the prompt
            image_paths: the paths to images to include in the message
            response_format: the format of the response
            prepend_string: the string to prepend to the message
        
        Returns:
            response: the response from the VLM
        """
        
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the message and response history.
        """
        
        raise NotImplementedError

    @abstractmethod
    def export(self, file_path: Path) -> None:
        """
        Export the message history to a file.

        Args:
            file_path: the path to the file to export
        """
        
        raise NotImplementedError
