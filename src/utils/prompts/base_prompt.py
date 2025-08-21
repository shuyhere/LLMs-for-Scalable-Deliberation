from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BasePrompt(ABC):
    """
    Base class for creating prompts with external information.
    
    This class provides a template for creating prompts that can incorporate
    external information and generate both system prompts and user inputs.
    """
    
    def __init__(self, external_info: Optional[Dict[str, Any]] = None):
        """
        Initialize the prompt with optional external information.
        
        Args:
            external_info: Dictionary containing external information to be
                          incorporated into the prompt
        """
        self.external_info = external_info or {}
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Generate the system prompt.
        
        Returns:
            The system prompt string
        """
        pass
    
    @abstractmethod
    def get_user_input(self) -> str:
        """
        Generate the user input prompt.
        
        Returns:
            The user input string
        """
        pass
    
    def get_prompt_pair(self) -> tuple[str, str]:
        """
        Get both system prompt and user input as a tuple.
        
        Returns:
            Tuple of (system_prompt, user_input)
        """
        return self.get_system_prompt(), self.get_user_input()
    
    def update_external_info(self, new_info: Dict[str, Any]) -> None:
        """
        Update the external information.
        
        Args:
            new_info: New external information to add/update
        """
        self.external_info.update(new_info)
    
    def get_external_info(self, key: str, default: Any = None) -> Any:
        """
        Get a specific piece of external information.
        
        Args:
            key: The key to retrieve
            default: Default value if key doesn't exist
            
        Returns:
            The value associated with the key, or default if not found
        """
        return self.external_info.get(key, default)


class SimplePrompt(BasePrompt):
    """
    Simple implementation of BasePrompt for basic use cases.
    """
    
    def __init__(self, user_template: str, system_template: Optional[str] = None, external_info: Optional[Dict[str, Any]] = None):
        """
        Initialize with template strings.
        
        Args:
            user_template: Template for user input
            system_template: Template for system prompt (defaults to "You are a helpful assistant")
            external_info: External information dictionary
        """
        super().__init__(external_info)
        self.user_template = user_template
        self.system_template = system_template or "You are a helpful assistant"
    
    def get_system_prompt(self) -> str:
        """Generate system prompt using the template and external info."""
        try:
            return self.system_template.format(**self.external_info)
        except KeyError as e:
            raise ValueError(f"Missing required external info key: {e}")
    
    def get_user_input(self) -> str:
        """Generate user input using the template and external info."""
        try:
            return self.user_template.format(**self.external_info)
        except KeyError as e:
            raise ValueError(f"Missing required external info key: {e}")


# Example usage:
if __name__ == "__main__":
    # Example 1: Simple prompt with default system prompt		
    
    user_template = "Please explain {topic} in simple terms."
    
    prompt = SimplePrompt(
        user_template=user_template,
        external_info={"topic": "neural networks"}
    )
    
    system_prompt, user_input = prompt.get_prompt_pair()
    print("System Prompt:", system_prompt)
    print("User Input:", user_input)
    
    # Example 2: Custom system prompt
    custom_prompt = SimplePrompt(
        user_template="Please explain {topic} in simple terms.",
        system_template="You are a helpful assistant with expertise in {domain}.",
        external_info={"domain": "machine learning", "topic": "neural networks"}
    )
    
    custom_system, custom_user = custom_prompt.get_prompt_pair()
    print("\nCustom System Prompt:", custom_system)
    print("Custom User Input:", custom_user)
    
    # Example 3: Update external info
    prompt.update_external_info({"topic": "deep learning"})
    new_user_input = prompt.get_user_input()
    print("\nUpdated User Input:", new_user_input)
