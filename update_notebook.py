import json

with open('notebooks/style_transfer_demo.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        for i, line in enumerate(cell['source']):
            if 'latest_checkpoint =' in line:
                cell['source'][i] = 'latest_checkpoint = "experiments/tacotron2-vae/tts_ptbr_fonetico_v4/checkpoints/epoch_42000"\n'
            # Also let's save the audio to a file
            if 'Audio(synthesized_audio' in line:
                cell['source'].append('\nimport torchaudio\n')
                cell['source'].append('torchaudio.save("style_transfer_output.wav", synthesized_audio.cpu(), 22050)\n')
                cell['source'].append('print("Saved output to style_transfer_output.wav")\n')

with open('notebooks/style_transfer_demo.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
