import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedDilatedBlock(nn.Module):
    def __init__(self, hidden_dim: int, dilation: int, dropout: float = 0.05):
        super().__init__()
        # A convolução gera o dobro dos canais para podermos dividir entre o Tanh (Filtro) e o Sigmoid (Porta)
        self.conv = nn.Conv1d(
            hidden_dim, hidden_dim * 2, 
            kernel_size=3, 
            padding=dilation, # Padding dinâmico acompanha a dilatação
            dilation=dilation
        )
        
        # Projeção para o próximo bloco
        self.res_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        # Projeção direta para a saída final (Skip Connection)
        self.skip_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        out = self.conv(x)
        
        # Gated Activation Unit (O segredo do WaveNet)
        # Divide os canais ao meio
        filter_act, gate_act = out.chunk(2, dim=1)
        # Filtro (informação) * Porta (passa ou não passa)
        out = torch.tanh(filter_act) * torch.sigmoid(gate_act)
        
        out = self.dropout(out)
        
        # Calcula o residual (vai para a próxima camada) e o skip (vai para o fim)
        res = self.res_conv(out)
        skip = self.skip_conv(out)
        
        # Soma a entrada com o residual multiplicada por uma constante de escala
        # para estabilizar a variância em redes muito profundas
        return (x + res) * 0.707, skip


class LatentResidualMapping(nn.Module):
    def __init__(self, channels: int = 80, hidden_dim: int = 512, num_blocks: int = 24):
        super().__init__()
        # 1. Camada de entrada
        self.input_layer = nn.Conv1d(channels, hidden_dim, kernel_size=3, padding=1)
        
        # 2. Pilha gigante de Blocos WaveNet
        self.blocks = nn.ModuleList()
        
        # Padrão de dilatações exponenciais repetido (1, 2, 4, 8, 16, 32, 64, 128)
        # Cria um campo de visão massivo sem explodir a VRAM.
        dilations = [1, 2, 4, 8, 16, 32, 64, 128]
        
        # Constrói os 24 blocos (3 ciclos completos da lista de dilatações)
        for i in range(num_blocks):
            dilation = dilations[i % len(dilations)]
            self.blocks.append(GatedDilatedBlock(hidden_dim, dilation=dilation))
            
        # 3. Camadas de Saída (agrega todos os skips)
        self.skip_proj1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.skip_proj2 = nn.Conv1d(hidden_dim, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_input = x
        
        # Entrada
        out = self.input_layer(x)
        
        # Passagem por todos os blocos e recolha dos skips
        skip_connections = []
        for block in self.blocks:
            out, skip = block(out)
            skip_connections.append(skip)
            
        # Soma global de todas as camadas
        # Isso garante que a rede não perde detalhe nenhum desde a primeira camada!
        out = sum(skip_connections)
        
        # Projeção final para 80 canais
        out = F.leaky_relu(out, 0.2)
        out = self.skip_proj1(out)
        out = F.leaky_relu(out, 0.2)
        out = self.skip_proj2(out)
        
        # Mapeamento Residual Global (Soma o espectrograma borrado com a máscara de correção gerada)
        return out + residual_input