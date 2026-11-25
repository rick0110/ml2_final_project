# Remote Access via SSH with Paramiko

Secure remote access to the PC using SSH with Paramiko, **with no need for administrator permission**.

## 📋 Features

- ✅ Key-based authentication (no password)
- ✅ No admin permission needed (port > 1024)
- ✅ RSA 4096-bit encryption
- ✅ Secure connections over SSH
- ✅ Python interface using Paramiko

## 🚀 Initial Setup

### Step 1: Run the setup
```bash
cd remote_access
python setup.py
```

This will:
1. Generate RSA 4096-bit SSH keys
2. Create the `authorized_keys` file
3. Configure the host key
4. Add your public key automatically

### Step 2: Start the server (on the machine to be accessed)
```bash
python ssh_server.py
```

The server will listen on port `2222` on all network interfaces.

### Step 3: Connect from the other PC

#### Option A: With the interactive client
```bash
python ssh_client.py
```

It will ask for:
- Hostname or IP of the server
- User (default: `user`)
- Port (default: `2222`)
- Path to the private key

#### Option B: With the system's standard SSH
```bash
ssh -i ~/.ssh/ml2_project -p 2222 user@SERVER_IP
```

#### Option C: From Python, programmatically
```python
from ssh_client import connect_to_server, execute_command

ssh = connect_to_server(
    hostname="192.168.1.100",
    private_key_path="~/.ssh/ml2_project",
    username="user"
)

stdout, stderr = execute_command(ssh, "whoami")
print(stdout)
ssh.close()
```

## 📁 Files

| File | Description |
|---------|-----------|
| `setup.py` | Initial setup script |
| `ssh_server.py` | SSH server |
| `ssh_client.py` | SSH client with an interface |
| `generate_keys.py` | SSH key generator |
| `config.py` | Settings |
| `.ssh/` | Directory with the keys (generated automatically) |

## 🔑 Security Details

### Generated Keys
- **Type**: RSA 4096-bit
- **Location**:
  - Private key: `~/.ssh/ml2_project` (permissions 600)
  - Public key: `~/.ssh/ml2_project.pub` (permissions 644)
  - Host key: `remote_access/.ssh/host_key`
  - Authorized keys: `remote_access/.ssh/authorized_keys`

### Port
- **Default**: 2222 (does not require admin)
- **Can be changed** in `config.py`

## 📝 Usage Examples

### Run a remote command
```bash
python ssh_client.py
# Select option 1 and type the command
```

### Interactive shell
```bash
python ssh_client.py
# Select option 2
```

### From Python
```python
from ssh_client import connect_to_server, execute_command

ssh = connect_to_server("192.168.1.100", "~/.ssh/ml2_project")

# Run several commands
stdout, _ = execute_command(ssh, "pwd")
print("Working directory:", stdout)

stdout, _ = execute_command(ssh, "ls -la")
print("Files:", stdout)

ssh.close()
```

## ⚙️ Configuration

Edit `config.py` to customize:
```python
SSH_PORT = 2222              # Port (> 1024 does not require admin)
TIMEOUT = 300                # Connection timeout
KEY_SIZE = 4096              # RSA key size
```

## 🛠️ Troubleshooting

### "Permission denied (publickey)"
- Check that the public key is in `authorized_keys`
- Check the permissions: `chmod 600 authorized_keys`

### "Address already in use"
- Another server instance is running
- Change the port in `config.py`

### "Connection refused"
- The server is not running
- Check that the port is correct
- Check the firewall

### "Connection timeout"
- The server is not responding
- Check the IP/hostname
- Check network connectivity

## 🔐 Security Considerations

✅ **Do**
- Keep the private key in a safe place
- Use strong passwords if needed
- Review `authorized_keys` regularly
- Use SSH with passwords only on a private network

❌ **Don't**
- Never share your private key
- Don't use well-known ports (< 1024)
- Don't expose the server directly to the internet without a firewall
- Don't commit private keys to git

## 📞 Support

For problems:
1. Check the server logs
2. Test connectivity with `ping`
3. Check that Paramiko is installed: `pip install paramiko`
4. Review the settings in `config.py`

## 📜 License

This code is provided as part of the ml2_final_project project.
