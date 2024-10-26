import os
import sys
sys.path.append("/home/grads/h/hasnat.md.abdullah/h2i_hatespeech_to_image/")

from diffusers import AutoPipelineForText2Image
import torch
import warnings
warnings.filterwarnings('ignore')

from PIL import Image
from data import super_set_neg_gender,super_set_neg_race,super_set_neg_disability,super_set_neg_lgbt
import shutil
# image generation parameters
NUMBER_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.0


def load_model (model_id):
    model = AutoPipelineForText2Image.from_pretrained(
    model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
  ).to("cuda")
    return model

def generate_image_sdxl(model, prompt, number_inference_steps = 30, guidance_scale = 7.0):
    image = model(
    prompt,
    num_inference_steps=number_inference_steps,
    guidance_scale= guidance_scale,
).images[0]
    return image
def dummy_image():
    return Image.new('RGB', (100, 100), color = 'red')
if __name__ == "__main__":

    # Load the model
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    model = load_model(model_id)

    
    output_dir = "/home/grads/h/hasnat.md.abdullah/h2i_hatespeech_to_image/data/generated_images/negative"
    filter_keywords = ['gender','disability']
    data_frames = [super_set_neg_gender,super_set_neg_disability]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filter_keyword, df in zip(filter_keywords, data_frames):
        print(f"Generating images for {filter_keyword}")
        filter_dir = os.path.join(output_dir, filter_keyword)
        if not os.path.exists(filter_dir):
            os.makedirs(filter_dir)

        for index, row in df.iterrows():
            print(f"Generating image for index: {index}")
            prompt = row['text']  # Assuming the dataframe has a column named 'text'
            try: 
                image_path = os.path.join(filter_dir, f"image_idx_{index}_sdxl.png")
                if os.path.exists(image_path):
                    print(f"Image already exists at {image_path}")
                    continue
                image = generate_image_sdxl(model, prompt, NUMBER_INFERENCE_STEPS, GUIDANCE_SCALE)
                image.save(image_path)
                print(f"Image saved at {image_path}")
                # flush gpu memory
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error generating image for {index}")
                print(e)
                continue
            
    # Create zip files for each folder
    for filter_keyword in filter_keywords:
        filter_dir = os.path.join(output_dir, filter_keyword)
        zip_path = os.path.join(output_dir, f"{filter_keyword}.zip")
        shutil.make_archive(zip_path.replace('.zip', ''), 'zip', filter_dir)
        print(f"Zip file created at {zip_path}")





    






