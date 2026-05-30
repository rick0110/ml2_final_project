import torch
from transformers import AutoProcessor, HubertModel

def load_hubert_emotion_model(freeze=True):
    
    processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960")
    
    # Carrega o HubertModel
    model = HubertModel.from_pretrained("facebook/hubert-large-ls960")
    
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
            
    return processor, model

