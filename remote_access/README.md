# Remote Access via SSH with Paramiko

Acesso remoto seguro ao PC usando SSH com Paramiko, **sem necessidade de permissão de administrador**.

## 📋 Características

- ✅ Autenticação baseada em chaves (sem senha)
- ✅ Sem necessidade de permissão de admin (porta > 1024)
- ✅ Encriptação RSA 4096-bit
- ✅ Conexões seguras via SSH
- ✅ Interface Python usando Paramiko

## 🚀 Setup Inicial

### Passo 1: Executar o setup
```bash
cd remote_access
python setup.py
```

Isso irá:
1. Gerar chaves SSH RSA 4096-bit
2. Criar arquivo `authorized_keys`
3. Configurar host key
4. Adicionar sua chave pública automaticamente

### Passo 2: Iniciar o servidor (na máquina a ser acessada)
```bash
python ssh_server.py
```

O servidor ficará escutando na porta `2222` em todas as interfaces de rede.

### Passo 3: Conectar do outro PC

#### Opção A: Via cliente interativo
```bash
python ssh_client.py
```

Ele vai pedir:
- Hostname ou IP do servidor
- Usuário (padrão: `user`)
- Porta (padrão: `2222`)
- Caminho da chave privada

#### Opção B: Via SSH padrão do sistema
```bash
ssh -i ~/.ssh/ml2_project -p 2222 user@SERVIDOR_IP
```

#### Opção C: Via Python programaticamente
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

## 📁 Arquivos

| Arquivo | Descrição |
|---------|-----------|
| `setup.py` | Script de setup inicial |
| `ssh_server.py` | Servidor SSH |
| `ssh_client.py` | Cliente SSH com interface |
| `generate_keys.py` | Gerador de chaves SSH |
| `config.py` | Configurações |
| `.ssh/` | Diretório com chaves (gerado automaticamente) |

## 🔑 Detalhes de Segurança

### Chaves Geradas
- **Tipo**: RSA 4096-bit
- **Localização**:
  - Chave privada: `~/.ssh/ml2_project` (permissões 600)
  - Chave pública: `~/.ssh/ml2_project.pub` (permissões 644)
  - Host key: `remote_access/.ssh/host_key`
  - Authorized keys: `remote_access/.ssh/authorized_keys`

### Porta
- **Padrão**: 2222 (não requer admin)
- **Pode ser alterada** em `config.py`

## 📝 Exemplos de Uso

### Executar comando remoto
```bash
python ssh_client.py
# Selecionar opção 1 e digitar comando
```

### Shell interativo
```bash
python ssh_client.py
# Selecionar opção 2
```

### Via Python
```python
from ssh_client import connect_to_server, execute_command

ssh = connect_to_server("192.168.1.100", "~/.ssh/ml2_project")

# Executar múltiplos comandos
stdout, _ = execute_command(ssh, "pwd")
print("Working directory:", stdout)

stdout, _ = execute_command(ssh, "ls -la")
print("Files:", stdout)

ssh.close()
```

## ⚙️ Configuração

Editar `config.py` para customizar:
```python
SSH_PORT = 2222              # Porta (> 1024 não requer admin)
TIMEOUT = 300                # Timeout da conexão
KEY_SIZE = 4096              # Tamanho da chave RSA
```

## 🛠️ Troubleshooting

### "Permission denied (publickey)"
- Verifique se a chave pública está em `authorized_keys`
- Verifique permissões: `chmod 600 authorized_keys`

### "Address already in use"
- Outra instância do servidor está rodando
- Mude a porta em `config.py`

### "Connection refused"
- Servidor não está rodando
- Verifique a porta correta
- Verifique firewall

### "Connection timeout"
- Servidor não está respondendo
- Verifique IP/hostname
- Verifique conectividade de rede

## 🔐 Considerações de Segurança

✅ **Faça**
- Guarde a chave privada em local seguro
- Use senhas fortes se necessário
- Revise `authorized_keys` regularmente
- Use SSH com senhas apenas em rede privada

❌ **Não faça**
- Nunca compartilhe sua chave privada
- Não use portas conhecidas (< 1024)
- Não exponha o servidor diretamente na internet sem firewall
- Não faça commit de chaves privadas no git

## 📞 Suporte

Para problemas:
1. Verifique logs do servidor
2. Teste conectividade com `ping`
3. Verifique se Paramiko está instalado: `pip install paramiko`
4. Revise configurações em `config.py`

## 📜 Licença

Este código é fornecido como parte do projeto ml2_final_project.
