from pathlib import Path
import subprocess
from tqdm import tqdm

# =========================
# CONFIGURAÇÕES
# =========================

INPUT_DIR = Path(
    "data/raw/libriSpeech-pt/mls_portuguese_opus"
)

OUTPUT_DIR = Path(
    "data/raw/libriSpeech-pt/mls_portuguese_wav"
)

TARGET_SR = 22050

# =========================
# CRIA PASTA DE SAÍDA
# =========================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# ENCONTRA TODOS OS .opus
# =========================

opus_files = list(INPUT_DIR.rglob("*.opus"))

print(f"Encontrados {len(opus_files)} arquivos .opus")

# =========================
# CONVERSÃO
# =========================

for opus_file in tqdm(opus_files):

    # caminho relativo
    relative_path = opus_file.relative_to(INPUT_DIR)

    # troca extensão
    wav_relative = relative_path.with_suffix(".wav")

    output_path = OUTPUT_DIR / wav_relative

    # cria subpastas
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # comando ffmpeg
    command = [
        "ffmpeg",
        "-y",
        "-i", str(opus_file),

        # mono
        "-ac", "1",

        # sample rate
        "-ar", str(TARGET_SR),

        # formato PCM 16 bits
        "-sample_fmt", "s16",

        str(output_path)
    ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

print("Conversão concluída.")