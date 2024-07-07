# agents.py

import json
from typing import List, Dict
from projects import Task

class Agent:
    def __init__(self, name: str, capabilities: List[str], prompts: Dict[str, str], model: str = None, **kwargs):
        self.name = name
        self.capabilities = capabilities
        self.prompts = prompts
        self.model = model
        self.settings = kwargs

    @classmethod
    def from_json(cls, json_path: str):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return cls(**data)

    def cancel_task(self, task: Task):
        """
        Cancels the execution of a task.

        Args:
            task: The task to cancel.
        """
        print(f"Canceling task: {task.name}")
        # TODO: Implement agent-specific cancellation logic