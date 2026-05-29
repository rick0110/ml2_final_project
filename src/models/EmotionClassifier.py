import torch
import torch.nn as nn
from src.models.HuBERT import load_hubert_model 

class HubertEmotionClassifier(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        
        self.processor, self.hubert = load_hubert_model(freeze=True)
        
        # O hubert-large geralmente tem uma dimensão oculta (hidden size) de 1024
        hidden_size = 1024 
        
        # 2. A SUA camada de classificação de emoções
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes) # num_classes será a quantidade de emoções do dataset
        )
        
    def forward(self, input_values, attention_mask=None):
        # Passa o áudio pelo HuBERT pegando os estados ocultos (output_hidden_states=True)
        outputs = self.hubert(
            input_values, 
            attention_mask=attention_mask, 
            output_hidden_states=True
        )
        
        # Pegamos a última camada oculta antes da tradução para texto
        # Formato: [batch_size, sequence_length, hidden_size]
        last_hidden_state = outputs.hidden_states[-1]
        
        # Mean Pooling: Reduz o tempo tirando a média do áudio inteiro
        # Formato vira: [batch_size, hidden_size]
        pooled_features = torch.mean(last_hidden_state, dim=1)
        
        # Passa as características extraídas pelo seu classificador de emoções
        logits = self.classifier(pooled_features)
        return logits