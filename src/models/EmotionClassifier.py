import torch
import torch.nn as nn
from src.models.HubertEmotionExtractor import load_hubert_emotion_model

class HubertEmotionClassifier(nn.Module):
    def __init__(self, num_classes=7):
        """
        num_classes: O número de emoções (no VERBO são 7: 
        Alegria, Desgosto, Medo, Neutro, Raiva, Surpresa, Tristeza)
        """
        super().__init__()
        
        self.processor, self.hubert = load_hubert_emotion_model(freeze=True)
        
        self.hidden_size = 768 

        #############################
        self.layer_weights = nn.Parameter(torch.ones(13))#
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.hidden_size, 256),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, input_values):
    
        outputs = self.hubert(input_values, output_hidden_states=True)

        hidden_states = outputs.hidden_states

        soft_weights = torch.softmax(self.layer_weights, dim=0)

        weighted_output = sum(w * h for w, h in zip(soft_weights, hidden_states))

        pooled_output = torch.mean(weighted_output, dim=1)
        
        #last_hidden_state = outputs.hidden_states[-5]
        
        #pooled_output = torch.mean(last_hidden_state, dim=1)
        
        logits = self.classifier(pooled_output)
        return logits  


