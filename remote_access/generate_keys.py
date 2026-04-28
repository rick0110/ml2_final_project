#!/usr/bin/env python3
"""
Generate SSH keys for remote access.
Creates RSA key pair without requiring admin permissions.
"""

import os
import paramiko
from pathlib import Path


def generate_ssh_keys(keys_dir="~/.ssh", key_name="ml2_project"):
    """
    Generate RSA SSH key pair.
    
    Args:
        keys_dir: Directory to store keys (default: ~/.ssh)
        key_name: Name of the key file (default: ml2_project)
    """
    keys_dir = Path(keys_dir).expanduser()
    keys_dir.mkdir(parents=True, exist_ok=True)
    
    private_key_path = keys_dir / key_name
    public_key_path = keys_dir / f"{key_name}.pub"
    
    # Check if keys already exist
    if private_key_path.exists() and public_key_path.exists():
        print(f"✓ Keys already exist at {private_key_path}")
        return str(private_key_path), str(public_key_path)
    
    print(f"Generating SSH keys in {keys_dir}...")
    
    # Generate RSA key (4096 bits for better security)
    key = paramiko.RSAKey.generate(bits=4096)
    
    # Save private key
    key.write_private_key_file(str(private_key_path))
    os.chmod(private_key_path, 0o600)
    print(f"✓ Private key saved to {private_key_path}")
    
    # Save public key in OpenSSH format
    with open(public_key_path, 'w') as f:
        f.write(f"ssh-rsa {key.get_base64()}\n")
    os.chmod(public_key_path, 0o644)
    print(f"✓ Public key saved to {public_key_path}")
    
    return str(private_key_path), str(public_key_path)


if __name__ == "__main__":
    private_key, public_key = generate_ssh_keys()
    print("\n" + "="*60)
    print("SSH Key Generation Complete!")
    print("="*60)
    print(f"Private Key: {private_key}")
    print(f"Public Key:  {public_key}")
    print("\nNext steps:")
    print("1. Keep the private key secure")
    print("2. Add the public key to authorized_keys on the server")
    print("3. Run ssh_server.py on the machine to be accessed")
    print("="*60)
