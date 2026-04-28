from setuptools import setup, find_packages

setup(
    name="prosody_style_transfer",
    version="0.1.0",
    description="Prosody and style transfer for Portuguese TTS using HuBERT + VITS",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "transformers>=4.35.0",
        "librosa>=0.10.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "soundfile>=0.12.0",
        "datasets>=2.14.0",
        "einops>=0.7.0",
    ],
)
