"""Remote access module for SSH connections using Paramiko."""

__version__ = "1.0.0"
__author__ = "ML2 Project"

from .ssh_client import connect_to_server, execute_command, interactive_shell
from .config import SSH_PORT, TIMEOUT

__all__ = [
    'connect_to_server',
    'execute_command',
    'interactive_shell',
    'SSH_PORT',
    'TIMEOUT'
]
