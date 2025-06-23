"""
Stub implementation of CovalOpenAIClient for local testing.
This is a simplified version that wraps the OpenAI client.
"""

import os
from openai import OpenAI


class CovalOpenAIClient:
    def __init__(self, api_key: str, manager=None):
        self.client = OpenAI(api_key=api_key)
        self.manager = manager

    @property
    def chat(self):
        return self.client.chat 