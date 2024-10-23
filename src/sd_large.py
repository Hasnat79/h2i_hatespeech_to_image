
# Gen
import os
import sys
sys.path.insert(0,'h2i_hatespeech_to_image/data')
from diffusers import StableDiffusion3Pipeline
import torch
import warnings
warnings.filterwarnings('ignore')
from huggingface_hub import login
from PIL import Image

from data import super_set_neg_gender,super_set_neg_race,super_set_neg_disability,super_set_neg_lgbt
# image generation parameters
NUMBER_INFERENCE_STEPS = 28
GUIDANCE_SCALE = 3.5
# def generate_image_sd_large(model, prompt, number_inference_steps, guidance_scale,output_path):
#     image = model(
#     "A capybara holding a sign that reads Hello World",
#     num_inference_steps=28,
#     guidance_scale=3.5,
# ).images[0]
# image.save(output_path)
def dummy_image():
    return Image.new('RGB', (100, 100), color = 'red')
if __name__ == "__main__":
    # login("HF_TOKEN")
    # Load the model
    # model = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16, cache_dir = "cache_dir/")
    # model = model.to("cuda")
    
    output_dir = "h2i_hatespeech_to_image/data/generated_images/negative"
    filter_keywords = ['gender','race','disability','lgbt']
    data_frames = [super_set_neg_gender,super_set_neg_race,super_set_neg_disability, super_set_neg_lgbt]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # The following section is for testing purposes, remove when running the actual image generation
    #------------------------------------------------
    i = 0
    #------------------------------------------------
    for filter_keyword, df in zip(filter_keywords, data_frames):
        print(f"Generating images for {filter_keyword}")
        filter_dir = os.path.join(output_dir, filter_keyword)
        if not os.path.exists(filter_dir):
            os.makedirs(filter_dir)

        for index, row in df.iterrows():
            print(f"Generating image for index: {index}")
            prompt = row['text']  # Assuming the dataframe has a column named 'text'
            # image = generate_image_sd_large(model, prompt, NUMBER_INFERENCE_STEPS, GUIDANCE_SCALE)
            try: 
                image = dummy_image()
                image_path = os.path.join(filter_dir, f"image_{index}_sdl.png")
                image.save(image_path)
                print(f"Image saved at {image_path}")
                # The following section is for testing purposes, remove when running the actual image generation
                #------------------------------------------------
                i+=1
                if i == 5:
                    i=0
                    break
                #------------------------------------------------
            except Exception as e:
                print(f"Error generating image for {index}")
                print(e)
                continue



