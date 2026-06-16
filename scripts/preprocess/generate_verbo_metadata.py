#!/usr/bin/env python3
"""
Script para gerar o metadata.csv do VERBO-Dataset.
Ele lê a pasta Audios, identifica a frase pelo ID no nome do arquivo 
e cria o mapeamento: nome_do_audio|texto_transcrito.
"""

import argparse
from pathlib import Path

# Tabela 1 do artigo corrigida (Mapeamento ID -> Frase)
VERBO_TEXTS = {
    "l1": "Os bombeiros estão equipados com uma arma.",
    "l2": "No próximo outono, Antônio vai a Minas em quinze de outubro.",
    "l3": "Agora vou pôr a camiseta e sair para uma caminhada.",
    "l4": "Um momento depois, ele caminhou ... e tropeçou.",
    "l5": "Eu queria o número de telefone de seu João.",
    "ns1": "A casa forte quer com pão.",
    "ns2": "A Força está para cima e alho vermelho.",
    "ns3": "O gato está rolando na pêra.",
    "ns4": "Salada de massa pata de carneiro amendoim.",
    "ns5": "Um quarenta e três vinte e sete noventa cinco mil.",
    "q1": "Sábado à noite, o que vai fazer?",
    "q2": "Você vai trazer aquela coisa com você?",
    "s1": "Os operários levantam cedo.",
    "s2": "A cachoeira faz muito barulho."
}

def main():
    parser = argparse.ArgumentParser(description="Gera o arquivo metadata.csv para o VERBO-Dataset")
    parser.add_argument(
        "--data_dir", 
        type=Path, 
        default=Path("data/raw/VERBO-Dataset"),
        help="Caminho para a pasta raiz do VERBO (onde a pasta Audios está)"
    )
    args = parser.parse_args()

    audios_dir = args.data_dir / "Audios"
    out_csv = args.data_dir / "metadata.csv"

    if not audios_dir.exists():
        print(f"❌ Erro: Pasta {audios_dir.resolve()} não encontrada!")
        print("Certifique-se de que os áudios estão dentro de 'data/raw/VERBO-Dataset/Audios/'")
        return

    wav_files = list(audios_dir.rglob("*.wav"))
    if not wav_files:
        print(f"❌ Erro: Nenhum arquivo .wav encontrado dentro de {audios_dir}")
        return

    matched_count = 0
    missing_count = 0

    print(f"🔍 Vasculhando {len(wav_files)} áudios em {audios_dir}...")

    with out_csv.open("w", encoding="utf-8") as f:
        for wav_path in wav_files:
            stem = wav_path.stem.lower()  # Nome do arquivo sem a extensão (ex: f1_neu_l1)
            parts = stem.split("_")
            
            audio_id = None
            
            # Tenta encontrar o ID do texto (l1, s2, ns1...) separando por "_"
            for part in parts:
                if part in VERBO_TEXTS:
                    audio_id = part
                    break
            
            # Fallback (caso os arquivos estejam nomeados sem "_")
            if not audio_id:
                for key in VERBO_TEXTS.keys():
                    if stem.endswith(key):
                        audio_id = key
                        break

            if audio_id:
                text = VERBO_TEXTS[audio_id]
                # Escreve no formato exato: nome_do_arquivo|texto
                f.write(f"{wav_path.stem}|{text}\n")
                matched_count += 1
            else:
                missing_count += 1
                print(f"⚠️ Aviso: Não foi possível identificar o ID da frase para o áudio: {wav_path.name}")

    print("\n✅ Processo concluído!")
    print(f"📄 Arquivo criado com sucesso em: {out_csv.resolve()}")
    print(f"📊 Áudios mapeados: {matched_count}")
    if missing_count > 0:
        print(f"⚠️ Áudios ignorados (ID não reconhecido): {missing_count}")

if __name__ == "__main__":
    main()