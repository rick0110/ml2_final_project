import torch
import os # Adicione o 'os' para criar a pasta de salvamento, se não existir
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data.verbo_dataset import VerboEmotionDataset
from src.models.EmotionClassifier import HubertEmotionClassifier

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Rodando o treinamento no dispositivo: {device}")

    epochs = 10
    batch_size = 4
    learning_rate = 1e-4

    print(" Carregando o modelo HuBERT para Emoções...")
    model = HubertEmotionClassifier(num_classes=7)
    model = model.to(device)

    print(" Mapeando os arquivos de áudio do VERBO...")
    dataset = VerboEmotionDataset(
        verbo_audios_dir="data/raw/verbo/Audios", 
        processor=model.processor
    )

    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    print(f"Tamanho do conjunto de treinamento: {train_size}")
    print(f"Tamanho do conjunto de validação: {val_size}"   )

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    os.makedirs("checkpoints", exist_ok=True)

    best_val_accuracy = 0.0
    
    print(" Iniciando o treinamento...")

    for epoch in range(epochs):
        print(f"\n{'-'*10} ÉPOCA {epoch+1}/{epochs} {'-'*10}")

        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0
    
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            outputs = model(inputs)
        
            loss = criterion(outputs, labels)
        
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            # Cálculo de acurácia no lote
            _, predicted = torch.max(outputs.data, 1) # Pega o índice da maior probabilidade
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = (train_correct / train_total) * 100
        
        # --- FASE DE VALIDAÇÃO ---
        model.eval() # Desativa o Dropout para a avaliação ser consistente
        val_loss = 0.0
        val_correct = 0
        val_total = 0 

        with torch.no_grad(): # Desativa o cálculo de gradientes (economiza memória e tempo)
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = (val_correct / val_total) * 100
        
        print(f"TREINO    | Loss: {avg_train_loss:.4f} | Acurácia: {train_accuracy:.2f}%")
        print(f"VALIDAÇÃO | Loss: {avg_val_loss:.4f} | Acurácia: {val_accuracy:.2f}%")

        if val_accuracy > best_val_accuracy:
            print(f" Novo recorde de Acurácia({best_val_accuracy:.2f}% -> {val_accuracy:.2f}%). Salvando modelo...")
            best_val_accuracy = val_accuracy
            
            # Salva apenas os pesos (state_dict), que é o padrão ouro do PyTorch
            torch.save(model.state_dict(), "checkpoints/melhor_modelo_emocoes.pth")

if __name__ == "__main__":
    train()

