import torch
import os
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, Subset
from src.data.verbo_dataset import VerboEmotionDataset
from src.models.EmotionClassifier import HubertEmotionClassifier


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Rodando o treinamento no dispositivo: {device}")

    epochs = 30
    batch_size = 4
    learning_rate = 3e-4

    print(" Carregando o modelo HuBERT para Emoções...")
    model = HubertEmotionClassifier(num_classes=7)
    model = model.to(device)

    print(" Mapeando os arquivos de áudio do VERBO...")
    dataset = VerboEmotionDataset(
        verbo_audios_dir="data/raw/verbo/Audios", 
        processor=model.processor
    )

    train_indices = []
    val_indices = []
    test_indices = []

    for idx, file_path in enumerate(dataset.file_paths):
        file_name = os.path.basename(file_path)
        try:
            parts = file_name.split('-')
            speaker = parts[1].lower() if len(parts) > 1 else ""
        except Exception:
            speaker = ""

        # Distribuição estrita de locutores para evitar vazamento de características físicas da voz
        if speaker in ['f6', 'm6']:
            test_indices.append(idx)    
        elif speaker in ['f5', 'm5']:
            val_indices.append(idx)     
        else:
            train_indices.append(idx)   

    print(f"   Treino (Atores f1-f4, m1-m4): {len(train_indices)} áudios")
    print(f"   Validação (Atores f5, m5):    {len(val_indices)} áudios")
    print(f"   Teste Final (Atores f6, m6):  {len(test_indices)} áudios")

    # Criação dos Sub-datasets isolados
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices) 

    # DataLoaders configurados corretamente
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    os.makedirs("checkpoints", exist_ok=True)

    best_val_accuracy = 0.0
    

    # Listas para armazenar o histórico do treinamento e gerar gráficos depois
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": []
    }

    print(" Iniciando o treinamento...")
    
    for epoch in range(epochs):
        print(f"\n{'-'*10} ÉPOCA {epoch+1}/{epochs} {'-'*10}")

        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0
    
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 🔥 MODIFICAÇÃO 1: Mapear e descobrir os caminhos originais dos arquivos de áudio do lote (batch) atual
            # Como o DataLoader embaralha os dados (shuffle=True), nós descobrimos de onde vieram resgatando os índices originais
            start_idx = batch_idx * batch_size
            end_idx = start_idx + inputs.size(0)
            
            # Recupera a lista de paths pertencentes aos índices deste lote específico
            batch_indices = train_indices[start_idx:end_idx]
            file_names = [dataset.file_paths[idx] for idx in batch_indices]
        
            optimizer.zero_grad()
        
            # 🔥 MODIFICAÇÃO 2: Passar os caminhos e os labels numéricos para dentro do forward do seu modelo
            outputs = model(inputs, file_names=file_names, labels=labels)
        
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
        model.eval() 
        val_loss = 0.0
        val_correct = 0
        val_total = 0 

        with torch.no_grad(): 
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

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_accuracy)
        history["val_acc"].append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            print(f" Novo recorde de Acurácia({best_val_accuracy:.2f}% -> {val_accuracy:.2f}%). Salvando modelo...")
            best_val_accuracy = val_accuracy
            
            # Salva apenas os pesos (state_dict), que é o padrão ouro do PyTorch
            torch.save(model.state_dict(), "checkpoints/melhor_modelo_emocoes.pth")
    
    # Carrega os pesos do modelo que teve a melhor performance na validação
    model.load_state_dict(torch.load("checkpoints/melhor_modelo_emocoes.pth"))
    model.eval()
    
    test_correct = 0
    test_total = 0

    # Listas para guardar o histórico de acertos e erros do gráfico
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            # Salva os palpites e os alvos reais convertendo para CPU/Numpy
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    final_test_accuracy = (test_correct / test_total) * 100
    print(f" ACURÁCIA FINAL DO MODELO (Teste): {final_test_accuracy:.2f}%")
    print("="*40)

    #   (Matriz de Confusão)
    # Nomes das classes na ordem correta do seu mapeamento (0 a 6)
    emotion_labels = ['Alegria', 'Desgosto', 'Medo', 'Neutro', 'Raiva', 'Surpresa', 'Tristeza']
    
    # Calcula a matriz matematicamente
    cm = confusion_matrix(all_labels, all_preds)
    
    # Desenha o gráfico estilizado (com tons de azul)
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=emotion_labels)
    
    # Plot do mapa de calor
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f"Matriz de Confusão - Teste Final (Acurácia: {final_test_accuracy:.2f}%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Salva a imagem no seu projeto
    plt.savefig("matriz_confusao_teste.png", dpi=300)
   
    # === Bloco para Gerar e Salvar os Gráficos ===
    epochs_range = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # Gráfico 1: Função de Perda (Loss)
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_loss"], label="Treino", color="blue", linewidth=2)
    plt.plot(epochs_range, history["val_loss"], label="Validação", color="red", linestyle="--")
    plt.title("Comportamento da Perda (Loss)")
    plt.xlabel("Épocas")
    plt.ylabel("Erro")
    plt.legend()
    plt.grid(True)

    # Gráfico 2: Acurácia (Accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["train_acc"], label="Treino", color="blue", linewidth=2)
    plt.plot(epochs_range, history["val_acc"], label="Validação", color="red", linestyle="--")
    plt.title("Evolução da Acurácia")
    plt.xlabel("Épocas")
    plt.ylabel("Acurácia (%)")
    plt.legend()
    plt.grid(True)

    # Salva a imagem na raiz do projeto
    plt.tight_layout()
    plt.savefig("historico_treinamento.png", dpi=300)
   
if __name__ == "__main__":
    train()


