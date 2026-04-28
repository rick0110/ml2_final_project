#!/usr/bin/env python3
"""
Quick start script for remote access.
Run this to test the SSH server and client.
"""

import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command."""
    print(f"\n{'='*60}")
    print(f"✓ {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"Error: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Quick start guide."""
    print("\n" + "="*60)
    print("Remote Access - Quick Start Guide")
    print("="*60)
    
    remote_access_dir = Path(__file__).parent
    
    print(f"\nProject directory: {remote_access_dir}")
    
    # Check setup
    setup_complete = False
    if (remote_access_dir / ".ssh" / "host_key").exists() and \
       (remote_access_dir / ".ssh" / "authorized_keys").exists():
        print("✓ Setup is complete!")
        setup_complete = True
    else:
        print("⚠ Setup not complete. Run: python setup.py")
    
    if not setup_complete:
        print("\nRun setup first:")
        print("  cd remote_access")
        print("  python setup.py")
        return
    
    # Show next steps
    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    
    print("\n1️⃣  ON THE MACHINE TO BE ACCESSED (Server):")
    print("   cd remote_access")
    print("   python ssh_server.py")
    print("   # Server will start listening on port 2222")
    
    print("\n2️⃣  GET THE SERVER'S IP ADDRESS:")
    print("   # Run this on the server machine:")
    print("   hostname -I   # Linux")
    print("   ipconfig      # Windows")
    print("   ifconfig      # macOS")
    
    print("\n3️⃣  ON THE CONNECTING MACHINE (Client):")
    print("   cd remote_access")
    print("   python ssh_client.py")
    print("   # Enter the server's IP address when prompted")
    
    print("\n4️⃣  ALTERNATIVE (Using SSH directly):")
    print("   ssh -i ~/.ssh/ml2_project -p 2222 user@SERVER_IP")
    
    # Show Python usage example
    print("\n" + "="*60)
    print("Python Usage Example:")
    print("="*60)
    print("""
from ssh_client import connect_to_server, execute_command

# Connect to server
ssh = connect_to_server(
    hostname="192.168.1.100",
    private_key_path="~/.ssh/ml2_project",
    username="user",
    port=2222
)

if ssh:
    # Execute command
    stdout, stderr = execute_command(ssh, "ls -la")
    print(stdout)
    
    # Close connection
    ssh.close()
""")
    
    print("="*60)
    print("Files created:")
    print("="*60)
    
    files = [
        ("ssh_server.py", "SSH server (run on machine to be accessed)"),
        ("ssh_client.py", "SSH client (run to connect)"),
        ("generate_keys.py", "SSH key generator"),
        ("setup.py", "Initial setup script"),
        ("config.py", "Configuration file"),
        (".ssh/host_key", "Server host key"),
        (".ssh/authorized_keys", "Authorized public keys"),
    ]
    
    for file, desc in files:
        file_path = remote_access_dir / file
        exists = "✓" if file_path.exists() else "✗"
        print(f"  {exists} {file:30} - {desc}")
    
    print("\n" + "="*60)
    print("Security Notes:")
    print("="*60)
    print("✓ Key-based authentication (no password)")
    print("✓ RSA 4096-bit encryption")
    print("✓ No admin permissions needed (port 2222)")
    print("✓ All communication is encrypted")
    print("✓ Private key location: ~/.ssh/ml2_project")
    print("✗ Don't share your private key!")
    print("✗ Don't commit private keys to git!")
    
    print("\n" + "="*60)
    print("Troubleshooting:")
    print("="*60)
    print("- Server not responding? Check firewall")
    print("- Connection refused? Is server running?")
    print("- Auth failed? Check ~/.ssh/ml2_project permissions")
    print("- Port already in use? Modify SSH_PORT in config.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
