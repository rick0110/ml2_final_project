# Scripts que geram os assets de docs/ (página GitHub Pages)

Ordem de execução (usa o .venv do projeto e GPU):

1. `eval_checkpoints.py` — reavalia os checkpoints dos dois modelos LJ numa
   amostra fixa de 32 falas (o log TensorBoard do baseline se perdeu) → `ckpt_eval.json`.
2. `generate_assets.py` — todos os áudios (style transfer, manipulação do latente,
   interpolação lerp/slerp) em `docs/assets/audio_wav/` + dados de análise em `analysis/`.
3. `plot_all.py` — todas as figuras em `docs/assets/img/` (lê `analysis/`,
   `ckpt_eval.json` e `tb_curves.json`).
4. Conversão p/ mp3: `for f in docs/assets/audio_wav/*.wav; do ffmpeg -i "$f" -ac 1 -q:a 5 docs/assets/audio/$(basename "${f%.wav}").mp3; done`

`tts_toolkit.py` é o módulo comum: carrega os 3 checkpoints (o baseline recebe
um patch para usar o último estado da GRU, já que o código atual usa atenção),
o WaveGlow, e implementa slerp/lerp e síntese a partir de z manipulado.
