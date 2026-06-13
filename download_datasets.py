import subprocess

comandos = [
    "python scripts/download_datasets/libriSpeech-pt",
    "chmod +x scripts/download_datasets/tts-portuguese-Corpora && ./scripts/download_datasets/tts-portuguese-Corpora",
    "python scripts/download_datasets/libriSpeech-en"
]

processos = []

print("🚀 Starting downloads in parallel...\n")

for cmd in comandos:
    processo = subprocess.Popen(cmd, shell=True)
    processos.append(processo)

print("✅ All processes were started in the background!")
print("The main code is free and the downloads are running together.\n")

# --- Opcional: Aguardar a finalização de todos os downloads ---
print("Waiting for all downloads to finish...")
for p in processos:
    p.wait()

print("\n🎉 All datasets were downloaded successfully!")
