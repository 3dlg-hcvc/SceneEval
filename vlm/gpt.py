import json
import yaml
import base64
from dataclasses import dataclass
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from .base import BaseVLM
from .registry import register_vlm

load_dotenv()
client = OpenAI()

# ----------------------------------------------------------------------------------------

@dataclass
class GPTConfig:
    """
    Configuration for the GPT interface.

    Attributes:
        model_name: the name of the GPT model to use
        prompt_file: the path to the prompt file
        seed: the seed passed to the API
    """

    model_name: str = "gpt-4o-2024-08-06"
    prompt_file: str = "./llm/prompts.yaml"
    seed: int | None = None

# ----------------------------------------------------------------------------------------

@register_vlm(config_class=GPTConfig)
class GPT(BaseVLM):
    def __init__(self, cfg: GPTConfig) -> None:
        """
        Initialize the GPT interface.
        
        Args:
            cfg: the configuration for the GPT interface
        """

        self.cfg = cfg
        
        # Initialize the OpenAI client and load the prompts
        with open(self.cfg.prompt_file) as file:
            self.prompts: dict[str, str] = yaml.safe_load(file)
        
        # Initialize the message history and response history
        self.message_history = [{"role": "system", "content": self.prompts["system"]}]
        self.response_history: list[str] = []
    
    def send(self,
             task: str,
             prompt_info: dict[str, str] | None = None,
             image_paths: list[Path] = [],
             response_format: BaseModel | None = None,
             prepend_string: str | None = None) -> BaseModel | str:
        """
        Send a message to the GPT model and return the response.

        Args:
            task: the task to perform
            prompt_info: the information to substitute into the prompt
            image_paths: the paths to images to include in the message
            response_format: the format of the response
            prepend_string: the string to prepend to the message
        
        Returns:
            response: the response from the GPT model
        """

        # Make the message
        text_message = self._make_message(task, prompt_info)
        if prepend_string:
            text_message = prepend_string + text_message
        print(f"\nGPT - Sending message to {self.cfg.model_name} with {len(image_paths)} images for task: {task}")
        print(f"Message:\n{text_message}")
        print()

        # Send the message
        self._send(text_message, image_paths, response_format)

        # Get the response from the message history
        response = self.response_history[-1]
        print("GPT - Received response")
        print(f"Response:\n{response}")
        print()

        return response
    
    def _make_message(self, task: str, prompt_info: dict[str, str] | None) -> str:
        """
        Make a prompt for a given task and substitute in the prompt information.

        Args:
            task: the task for which to make the prompt
            prompt_info: the information to substitute into the prompt
        
        Returns:
            prompt: the prompt for the task
        """

        prompt = self.prompts[task]
        if prompt_info:
            for key, value in prompt_info.items():
                prompt = prompt.replace(f"<{key.upper()}>", value)

        return prompt

    def _send(self,
              text_content: str,
              image_paths: list[Path] = [],
              response_format: BaseModel | None = None) -> None:
        """
        Send a message to the GPT model and store the response.

        Args:
            text_content: the text content of the message
            image_paths: the paths to images to include in the message
            response_format: the format of the response
        """

        # Create the message with the text content
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": text_content}
            ]
        }

        # Add the images to the message if any
        for image_path in image_paths:
            with open(image_path, "rb") as file:
                base64_image = base64.b64encode(file.read()).decode("utf-8")

            message["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "low"
                }
            })
        
        # Store the message into the message history
        self.message_history.append(message)
        
        # Send the message
        payload = {
            "model": self.cfg.model_name,
            "messages": self.message_history,
            "seed": self.cfg.seed
        }
        if response_format:
            payload["response_format"] = response_format
        # completion = client.chat.completions.create(**payload)
        completion = client.beta.chat.completions.parse(**payload)
        
        # Store the response
        # response = completion.choices[0].message.content
        response_message = completion.choices[0].message
        if response_format:
            if response_message.parsed:
                response: BaseModel = response_message.parsed
            else:
                response: str = response_message.refusal
        else:
            response: str = response_message.content

        self.message_history.append({"role": "assistant", "content": [{"type": "text", "text": response_message.content}]})
        self.response_history.append(response)
    
    def reset(self) -> None:
        """
        Reset the message and response history.
        """

        self.message_history = [{"role": "system", "content": [{"type": "text", "text": self.prompts["system"]}]}]
        self.response_history = []
    
    def export(self, file_path: Path) -> None:
        """
        Export the message history to a json file.

        Args:
            file_path: the path to the file to export
        """

        messages = self.message_history
        
        # Remove images from the messages
        for message in messages:
            if message["role"] == "user" and type(message["content"]) == list:
                old_length = len(message["content"])
                message["content"] = [
                    content for content in message["content"] if content["type"] != "image_url"
                ]
                num_images = old_length - len(message["content"])
                message["content"].append(f"Had {num_images} images attached.")

        with open(file_path, "w") as file:
            json.dump(messages, file, indent=4)
