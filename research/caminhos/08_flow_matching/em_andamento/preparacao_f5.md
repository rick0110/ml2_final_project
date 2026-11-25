# Preparação do F5-TTS (pausado)

**Início:** 23/11/2026 · **Caminho:** [08](../README.md)

## Feito

- Ambiente instalado numa das máquinas do cluster (pasta `~/f5/`): `f5-tts` 1.1.22 e código do F5-TTS clonado, com as configurações em `src/f5_tts/configs/`.
- Dados baixados do LibriTTS-R (versão restaurada do LibriTTS, em parquet): train-clean-100, train-clean-360 e test-clean, cerca de 41 GB.

## Falta

1. Converter os parquet para o formato do F5-TTS. O `prepare_libritts.py` do F5 espera pastas com wav e `.normalized.txt`, então precisa de uma adaptação.
2. Montar subconjuntos com N variável e o mesmo total de horas.
3. Configurações pequenas em 2–3 tamanhos; medir velocidade e memória numa GPU livre.
4. Rodar a curva com a avaliação de bits de voz.

## Por que pausou

As GPUs estão ocupadas com os experimentos do Tacotron, que respondem às perguntas dos caminhos 01–04 mais depressa. Retomar quando uma máquina puder ficar dedicada.

## Previsões

Ainda não registradas: registrar antes do primeiro treino.
