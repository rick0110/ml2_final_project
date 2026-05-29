import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data.verbo_dataset import VerboEmotionDataset
from src.models.EmotionClassifier import HubertEmotionClassifier

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Rodando o treinamento no dispositivo: {device}")

    print(" Carregando o modelo HuBERT para Emoções...")
    model = HubertEmotionClassifier(num_classes=7)
    model = model.to(device)

    print(" Mapeando os arquivos de áudio do VERBO...")
    dataset = VerboEmotionDataset(
        verbo_audios_dir="data/raw/verbo/Audios", 
        processor=model.processor
    )
    
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    print(" Iniciando teste do motor de treinamento...")
    model.train()
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        if batch_idx % 5 == 0:
            print(f"Lote [{batch_idx}/{len(train_loader)}] | Erro (Loss): {loss.item():.4f}")
            

if __name__ == "__main__":
    train()