import os
from PIL import Image
import io
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_images_from_tfevents(log_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the event accumulator
    event_acc = EventAccumulator(log_dir, size_guidance={'images': 0}) # 0 means load all
    event_acc.Reload()
    
    # Get all image tags
    image_tags = event_acc.Tags()['images']
    print(f"Found image tags: {image_tags}")
    
    for tag in image_tags:
        events = event_acc.Images(tag)
        safe_tag = tag.replace('/', '_')
        for event in events:
            step = event.step
            # event.encoded_image_string contains the raw bytes of the image
            try:
                img = Image.open(io.BytesIO(event.encoded_image_string))
                img_name = f"{safe_tag}_step{step}.png"
                img_path = os.path.join(output_dir, img_name)
                img.save(img_path)
                print(f"Saved {img_path}")
            except Exception as e:
                print(f"Error saving image {tag} at step {step}: {e}")

print("Extracting from libri_pt_br_fonetico_v4")
extract_images_from_tfevents(
    "experiments/tacotron2-vae/libri_pt_br_fonetico_v4/logs",
    "experiments/tacotron2-vae/libri_pt_br_fonetico_v4/extracted_images"
)

print("Extracting from tts_ptbr_fonetico_v4")
extract_images_from_tfevents(
    "experiments/tacotron2-vae/tts_ptbr_fonetico_v4/logs",
    "experiments/tacotron2-vae/tts_ptbr_fonetico_v4/extracted_images"
)
