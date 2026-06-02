import torch
from transformers import AutoFeatureExtractor, HubertModel

def load_hubert_emotion_model(freeze=True):
    
    processor = AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    
    # Carrega o HubertModel
    model = HubertModel.from_pretrained("facebook/hubert-base-ls960", use_safetensors=True)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False
            
    return processor, model


