#!/usr/bin/env python3
"""
SSH Server using Paramiko.
Allows remote access to this machine without admin permissions.
Uses key-based authentication for security.
"""

import os
import socket
import threading
import paramiko
import sys
from pathlib import Path
from config import SSH_PORT, TIMEOUT, HOST_KEY_NAME, AUTHORIZED_KEYS_DIR


class SSHServer(paramiko.ServerInterface):
    """SSH Server implementation using Paramiko."""
    
    def __init__(self, authorized_keys_file):
        self.authorized_keys_file = authorized_keys_file
        self.event = threading.Event()
    
    def check_auth_password(self, username, password):
        """Disable password authentication."""
        return paramiko.AUTH_FAILED
    
    def check_auth_publickey(self, username, key):
        """Verify public key authentication."""
        try:
            with open(self.authorized_keys_file, 'r') as f:
                authorized_keys = f.readlines()
            
            client_key_str = f"ssh-rsa {key.get_base64()}"
            
            for line in authorized_keys:
                if line.strip() == client_key_str.strip():
                    print(f"✓ Public key authentication successful for {username}")
                    return paramiko.AUTH_SUCCESSFUL
            
            print(f"✗ Public key not authorized for {username}")
            return paramiko.AUTH_FAILED
        
        except FileNotFoundError:
            print(f"✗ Authorized keys file not found: {self.authorized_keys_file}")
            return paramiko.AUTH_FAILED
    
    def check_channel_request(self, kind, chanid):
        """Accept channel requests."""
        if kind == 'session':
            return paramiko.OPEN_SUCCEEDED
        return paramiko.OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED
    
    def check_channel_shell_request(self, channel):
        """Accept shell requests."""
        self.event.set()
        return paramiko.OPEN_SUCCEEDED
    
    def get_allowed_auths(self, username):
        """Specify allowed authentication methods."""
        return 'publickey'


def setup_host_key(keys_dir=None):
    """Setup or load host key for the SSH server."""
    if keys_dir is None:
        keys_dir = Path(__file__).parent / ".ssh"
    else:
        keys_dir = Path(keys_dir)
    
    keys_dir.mkdir(parents=True, exist_ok=True)
    host_key_path = keys_dir / HOST_KEY_NAME
    
    if host_key_path.exists():
        print(f"✓ Loading existing host key from {host_key_path}")
        host_key = paramiko.RSAKey.from_private_key_file(str(host_key_path))
    else:
        print(f"Generating new host key...")
        host_key = paramiko.RSAKey.generate(bits=4096)
        host_key.write_private_key_file(str(host_key_path))
        os.chmod(host_key_path, 0o600)
        print(f"✓ Host key saved to {host_key_path}")
    
    return host_key


def setup_authorized_keys(keys_dir=None):
    """Setup authorized_keys directory."""
    if keys_dir is None:
        keys_dir = Path(__file__).parent / ".ssh"
    else:
        keys_dir = Path(keys_dir)
    
    keys_dir.mkdir(parents=True, exist_ok=True)
    authorized_keys_file = keys_dir / "authorized_keys"
    
    if not authorized_keys_file.exists():
        authorized_keys_file.touch()
        os.chmod(authorized_keys_file, 0o600)
        print(f"✓ Created authorized_keys file at {authorized_keys_file}")
    
    return str(authorized_keys_file)


def handle_client(client, addr, host_key, authorized_keys_file):
    """Handle individual SSH client connection."""
    print(f"\nIncoming connection from {addr}")
    
    try:
        transport = paramiko.Transport(client)
        transport.add_server_key(host_key)
        transport.set_subsystem_handler(
            'sftp',
            paramiko.SFTPServer,
            paramiko.ServerInterface()
        )
        
        server = SSHServer(authorized_keys_file)
        transport.start_server(server=server)
        
        # Keep connection alive
        channel = transport.accept(timeout=TIMEOUT)
        if channel is None:
            print(f"✗ No channel opened from {addr}")
        else:
            print(f"✓ Channel opened from {addr}")
            channel.close()
        
        transport.close()
    
    except Exception as e:
        print(f"✗ Error handling client {addr}: {e}")
    
    finally:
        client.close()


def start_ssh_server(host='0.0.0.0', port=SSH_PORT, keys_dir=None):
    """
    Start SSH server listening for connections.
    
    Args:
        host: Host to bind to (default: 0.0.0.0 - all interfaces)
        port: Port to listen on (default: 2222)
        keys_dir: Directory for SSH keys (default: remote_access/.ssh)
    """
    print("="*60)
    print("SSH Server Starting")
    print("="*60)
    
    # Setup keys
    host_key = setup_host_key(keys_dir)
    authorized_keys_file = setup_authorized_keys(keys_dir)
    
    print(f"\nListening on {host}:{port}")
    print(f"Authorized keys: {authorized_keys_file}")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    # Create listening socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(5)
    
    try:
        while True:
            try:
                client, addr = sock.accept()
                # Handle each client in a separate thread
                thread = threading.Thread(
                    target=handle_client,
                    args=(client, addr, host_key, authorized_keys_file)
                )
                thread.daemon = True
                thread.start()
            
            except KeyboardInterrupt:
                break
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        sock.close()
        print("\nSSH Server stopped.")


if __name__ == "__main__":
    keys_dir = Path(__file__).parent / ".ssh"
    start_ssh_server(keys_dir=keys_dir)
