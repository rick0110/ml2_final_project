#!/usr/bin/env python3
"""
Setup script for remote access.
Initializes SSH keys and authorized_keys.
"""

import sys
import os
from pathlib import Path
from generate_keys import generate_ssh_keys
from ssh_server import setup_host_key, setup_authorized_keys


def setup_remote_access():
    """
    Complete setup for remote access.
    """
    print("="*60)
    print("Remote Access Setup")
    print("="*60)
    
    # Setup directory
    remote_access_dir = Path(__file__).parent
    ssh_dir = remote_access_dir / ".ssh"
    
    print(f"\nSetup directory: {remote_access_dir}")
    print(f"SSH keys directory: {ssh_dir}\n")
    
    # Step 1: Generate user SSH keys
    print("Step 1: Generate SSH keys")
    print("-" * 60)
    private_key, public_key = generate_ssh_keys(
        keys_dir=Path.home() / ".ssh",
        key_name="ml2_project"
    )
    
    # Step 2: Setup host key
    print("\nStep 2: Setup host key")
    print("-" * 60)
    host_key = setup_host_key(ssh_dir)
    
    # Step 3: Setup authorized_keys
    print("\nStep 3: Setup authorized keys")
    print("-" * 60)
    authorized_keys_file = setup_authorized_keys(ssh_dir)
    
    # Step 4: Add public key to authorized_keys
    print("\nStep 4: Add public key to authorized keys")
    print("-" * 60)
    try:
        with open(public_key, 'r') as f:
            public_key_content = f.read()
        
        authorized_keys_path = Path(authorized_keys_file)
        existing_keys = set()
        
        if authorized_keys_path.exists():
            with open(authorized_keys_path, 'r') as f:
                existing_keys = set(f.readlines())
        
        if public_key_content.strip() not in [k.strip() for k in existing_keys]:
            with open(authorized_keys_path, 'a') as f:
                f.write(public_key_content)
            print(f"✓ Added public key to {authorized_keys_file}")
        else:
            print(f"✓ Public key already in {authorized_keys_file}")
    
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Summary
    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    print(f"\nPrivate key: {private_key}")
    print(f"Public key: {public_key}")
    print(f"Host key: {remote_access_dir / '.ssh' / 'host_key'}")
    print(f"Authorized keys: {authorized_keys_file}")
    
    print("\nNext steps:")
    print("1. Run 'python ssh_server.py' on the machine to be accessed")
    print("2. Get the IP address of that machine")
    print("3. Run 'python ssh_client.py' on the connecting machine")
    print("4. Enter the IP/hostname and connect!")
    
    print("\nNOTE: You need admin permissions only for port < 1024")
    print("      Using port 2222 (> 1024) doesn't require admin")
    print("="*60)
    
    return True


if __name__ == "__main__":
    try:
        setup_remote_access()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Setup failed: {e}")
        sys.exit(1)
