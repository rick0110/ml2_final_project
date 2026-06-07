import torch
import torch.nn as nn
import torchaudio.transforms as T
import os
import matplotlib.pyplot as plt
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
        self.layer_weights = nn.Parameter(torch.ones(13))

        # configurar o mel espectrograma
        self.n_mels = 128
        self.mel_transform = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=512,
            n_mels=self.n_mels
        )

        self.save_dir = "data/mel_spectrograms_visual"
        os.makedirs(self.save_dir, exist_ok=True)

        self.saved_images_count = 0
        self.max_images_to_save = 100

        self.emotion_map = {
            0: "Alegria", 1: "Desgosto", 2: "Medo", 
            3: "Neutro", 4: "Raiva", 5: "Surpresa", 6: "Tristeza"
        }

        ##############################

        self.combined_size = self.hidden_size + self.n_mels

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.combined_size, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_values, file_names=None, labels=None):

        outputs = self.hubert(input_values, output_hidden_states=True)

        hidden_states = outputs.hidden_states

        soft_weights = torch.softmax(self.layer_weights, dim=0)

        weighted_output = sum(w * h for w, h in zip(soft_weights, hidden_states))

        pooled_output = torch.mean(weighted_output, dim=1)
        
        #last_hidden_state = outputs.hidden_states[-5]
        
        #pooled_output = torch.mean(last_hidden_state, dim=1)

        mel_spec = self.mel_transform(input_values)
        log_mel_spec = torch.log(mel_spec + 1e-6)

        if self.training and self.saved_images_count < self.max_images_to_save:
            # Percorre o lote (batch) atual
            for i in range(log_mel_spec.size(0)):
                if self.saved_images_count >= self.max_images_to_save:
                    break

                spec_numpy = log_mel_spec[i].detach().cpu().numpy()
                
                audio_name = "Desconhecido"
                emotion_name = "Nao_Identificada"
                
                if file_names is not None and i < len(file_names):
                    audio_name = os.path.basename(file_names[i]) # Pega apenas o nome (ex: m1-raiva-1.wav)
                    
                if labels is not None and i < len(labels):
                    lbl_idx = labels[i].item() # Converte o tensor numérico para int do Python
                    emotion_name = self.emotion_map.get(lbl_idx, "Desconhecido")
                
                # Desenha o espectrograma usando Matplotlib
                plt.figure(figsize=(9, 5))
                plt.imshow(spec_numpy, aspect='auto', origin='lower', cmap='viridis')
                
                #  Título da imagem atualizado para os seus slides
                plt.title(f"Mel Spectrogram\nArquivo: {audio_name} | Emoção Real: {emotion_name}", fontsize=11, fontweight='bold')
                plt.xlabel("Tempo (Frames)")
                plt.ylabel("Filtros Mel (Frequência)")
                plt.colorbar(format='%+2.0f dB')
                plt.tight_layout()
                
                # Nome do arquivo final categorizado por Emoção + Nome Original
                clean_filename = f"{emotion_name}_{audio_name}.png"
                plt.savefig(os.path.join(self.save_dir, clean_filename))
                plt.close() # Fecha a imagem para liberar memória RAM
                
                self.saved_images_count += 1

        # Pooling temporal do Mel para enviar ao classificador
        mel_pooled = torch.mean(log_mel_spec, dim=2) 

        #PARTE 3: FUSÃO E CLASSIFICAÇÃO 
        combined_features = torch.cat((pooled_output, mel_pooled), dim=1) 
        logits = self.classifier(combined_features)
        return logits


