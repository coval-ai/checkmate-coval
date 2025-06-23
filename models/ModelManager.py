"""
Stub implementation of ModelManager base class for local testing.
"""

import json
import logging
from typing import Dict, Any


class ModelManager:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.type = "unknown"
        self.ready = False
        self._test_cases = {}
        self._simulation_outputs = {}
        self.logger = logging.getLogger(__name__)

    def _print_and_log(self, message: str):
        """Print and log a message"""
        print(message)
        self.logger.info(message)

    def _start(self, config: Dict[str, Any]):
        """Initialize the model manager"""
        raise NotImplementedError("Subclasses must implement _start")

    def get_input_str_from_test_case_id(self, test_case_id: str) -> str:
        """Get input string from test case ID"""
        test_case = self._test_cases.get(test_case_id)
        if not test_case:
            raise ValueError(f"Test case {test_case_id} not found")
        return test_case.get("input", "")

    def _populate_simulation_output(self, simulation_output_id: str, output: str, status: str):
        """Store simulation output"""
        self._simulation_outputs[simulation_output_id] = {
            "output": output,
            "status": status
        }

    def run_simulation(self, test_case_id: str, simulation_output_id: str) -> bool:
        """Run the simulation"""
        raise NotImplementedError("Subclasses must implement run_simulation") 