"""
Configuration for SSH remote access.
"""

# SSH Server Configuration
SSH_PORT = 2222                    # Port to listen on (>1024 doesn't need admin)
TIMEOUT = 300                      # Connection timeout in seconds
HOST_KEY_NAME = "host_key"         # Name of host key file
AUTHORIZED_KEYS_DIR = ".ssh"       # Directory for keys (relative to remote_access folder)

# Client Configuration
DEFAULT_USERNAME = "user"
DEFAULT_HOSTNAME = "localhost"

# Security
KEY_SIZE = 4096                    # RSA key size in bits
MAX_CONNECTIONS = 5               # Maximum concurrent connections

# Logging
LOG_LEVEL = "INFO"                 # DEBUG, INFO, WARNING, ERROR
ENABLE_LOGGING = True
