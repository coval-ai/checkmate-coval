"""
Stub implementation of StatusCodes for local testing.
"""

from enum import Enum
from dataclasses import dataclass


class StatusCode(Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class Status:
    code: StatusCode
    message: str 