#!/usr/bin/env python3
"""
SSH Client using Paramiko.
Connects to the SSH server for remote access.
"""

import paramiko
import sys
from pathlib import Path
from config import SSH_PORT, TIMEOUT


def connect_to_server(hostname, private_key_path, username="user", port=SSH_PORT):
    """
    Connect to SSH server using key-based authentication.
    
    Args:
        hostname: Hostname or IP address of the server
        private_key_path: Path to private SSH key
        username: Username for login (default: user)
        port: SSH port (default: 2222)
    
    Returns:
        SSH client object if successful, None otherwise
    """
    print(f"Connecting to {hostname}:{port} as {username}...")
    
    # Check if private key exists
    private_key_path = Path(private_key_path).expanduser()
    if not private_key_path.exists():
        print(f"✗ Private key not found: {private_key_path}")
        return None
    
    try:
        # Create SSH client
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Load private key
        private_key = paramiko.RSAKey.from_private_key_file(str(private_key_path))
        
        # Connect to server
        ssh.connect(
            hostname=hostname,
            port=port,
            username=username,
            pkey=private_key,
            timeout=TIMEOUT,
            look_for_keys=False,
            allow_agent=False
        )
        
        print(f"✓ Connected successfully to {hostname}")
        return ssh
    
    except paramiko.AuthenticationException:
        print(f"✗ Authentication failed. Check if your public key is in authorized_keys")
        return None
    
    except paramiko.SSHException as e:
        print(f"✗ SSH error: {e}")
        return None
    
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def execute_command(ssh, command):
    """
    Execute command on remote server.
    
    Args:
        ssh: SSH client object
        command: Command to execute
    
    Returns:
        Tuple of (stdout, stderr)
    """
    try:
        stdin, stdout, stderr = ssh.exec_command(command)
        stdout_data = stdout.read().decode('utf-8')
        stderr_data = stderr.read().decode('utf-8')
        return stdout_data, stderr_data
    
    except Exception as e:
        print(f"✗ Error executing command: {e}")
        return None, str(e)


def interactive_shell(ssh):
    """
    Start interactive shell session.
    
    Args:
        ssh: SSH client object
    """
    try:
        channel = ssh.invoke_shell()
        print("\n✓ Interactive shell started (type 'exit' to quit)")
        print("="*60 + "\n")
        
        import sys
        import select
        
        while True:
            # Check for user input
            readable, _, _ = select.select([sys.stdin, channel], [], [], 0.1)
            
            if sys.stdin in readable:
                user_input = sys.stdin.read(1)
                if user_input:
                    channel.send(user_input)
            
            if channel in readable:
                try:
                    output = channel.recv(1024)
                    if output:
                        sys.stdout.write(output.decode('utf-8'))
                    else:
                        break
                except:
                    break
    
    except Exception as e:
        print(f"✗ Error in interactive shell: {e}")
    
    finally:
        channel.close()


def main():
    """Main client interface."""
    print("="*60)
    print("SSH Client")
    print("="*60)
    
    # Get connection details
    hostname = input("\nEnter hostname or IP: ").strip()
    if not hostname:
        print("Hostname required!")
        return
    
    username = input("Enter username [user]: ").strip() or "user"
    
    port_input = input("Enter port [2222]: ").strip()
    port = int(port_input) if port_input else SSH_PORT
    
    home = Path.home()
    default_key = home / ".ssh" / "ml2_project"
    private_key = input(f"Enter private key path [{default_key}]: ").strip() or str(default_key)
    
    # Connect to server
    ssh = connect_to_server(hostname, private_key, username, port)
    if not ssh:
        return
    
    # Interactive menu
    try:
        while True:
            print("\n" + "="*60)
            print("Options:")
            print("  1. Execute command")
            print("  2. Interactive shell")
            print("  3. Disconnect")
            print("="*60)
            
            choice = input("Select option [1-3]: ").strip()
            
            if choice == "1":
                command = input("Enter command: ").strip()
                if command:
                    stdout, stderr = execute_command(ssh, command)
                    if stdout:
                        print(f"\nOutput:\n{stdout}")
                    if stderr:
                        print(f"\nError:\n{stderr}")
            
            elif choice == "2":
                interactive_shell(ssh)
            
            elif choice == "3":
                print("Disconnecting...")
                break
            
            else:
                print("Invalid option!")
    
    finally:
        ssh.close()
        print("✓ Disconnected")


if __name__ == "__main__":
    main()
