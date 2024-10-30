from huggingface_hub import login
import os
import sys
grace_path = "/scratch/user/hasnat.md.abdullah/h2i_hatespeech_to_image/"
spiderman_path = "/home/grads/h/hasnat.md.abdullah/h2i_hatespeech_to_image/"
sys.path.append(grace_path)
sys.path.append(spiderman_path)
from PIL import Image

import torch
import diffusers
import transformers
import accelerate
print("Torch Version",torch.__version__)
print("Diffusers Version",diffusers.__version__)
print("Transformers Version",transformers.__version__)
print("Accelerate Version",accelerate.__version__)

import warnings
warnings.filterwarnings('ignore')
from PIL import Image
from data import super_set_hateful_gender, super_set_hateful_disability
import shutil
import argparse


from diffusers import StableDiffusion3Pipeline

# image generation parameters
NUMBER_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.0

def load_model (model_id):
    
    model = StableDiffusion3Pipeline.from_pretrained( model_id, cache_dir="cache")
    model.enable_model_cpu_offload()
    return model
def generate_image_sd_large(model, prompt, number_inference_steps = 30, guidance_scale = 7.0):
    image = model(
    prompt,
    num_inference_steps=number_inference_steps,
    guidance_scale= guidance_scale,
    ).images[0]
    return image
def dummy_image():
    image = Image.new('RGB', (224, 224), color = 'red')
    return image

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images using Stable Diffusion")
    parser.add_argument('--start_idx', type=int, required=True, help='Start index of the dataframe')
    parser.add_argument('--end_idx', type=int, required=True, help='End index of the dataframe')
    return parser.parse_args()


if __name__ == "__main__":
    # args = parse_args()
    # start_idx = args.start_idx
    # end_idx = args.end_idx
    # Load the model
    model_id = "stabilityai/stable-diffusion-3.5-large"
    model = load_model(model_id)

    # hateful image generation directory
    output_dir = "/scratch/user/hasnat.md.abdullah/h2i_hatespeech_to_image/data/generated_images/sdl_3_5/hateful/"
    # filter keywords
    # filter_keywords = ['gender']
    filter_keywords = ['disability']
    # uncomment this for negative
    data_frames = [super_set_hateful_gender, super_set_hateful_disability]
    # verify
    print(data_frames[0].shape)
    print(data_frames[1].shape)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # for each prompt n_pass images will be generated
    n_pass = 5
    for filter_keyword, df in zip(filter_keywords, data_frames):
        print(f"Generating images for {filter_keyword}")
        filter_dir = os.path.join(output_dir, filter_keyword)
        if not os.path.exists(filter_dir):
            os.makedirs(filter_dir)

        for index, row in df.iterrows():
            # index start from 0,1,2,3...
            print(f"Generating image for index: {index}")
            prompt = row['text']  # Assuming the dataframe has a column named 'text'
            # creates a directory for each index
            dir_path = os.path.join(filter_dir, f"{index}")
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            # check if the directory already has 5 images, if yes then skip
            if len([name for name in os.listdir(dir_path) if name.endswith('.png')]) == 5:
                print(f"Directory {dir_path} already has 5 images, skipping...")
                continue
            # generate n_pass images for each prompt and save them in the directory: img_i.png
            for i in range(n_pass):

                try: 
                    image_path = os.path.join(dir_path, f"img_{i+1}.png")
                    if os.path.exists(image_path):
                        print(f"Image already exists at {image_path}")
                        continue
                    image = generate_image_sd_large(model, prompt, NUMBER_INFERENCE_STEPS, GUIDANCE_SCALE)
                    # print("dummy image generated")
                    # image = dummy_image()
                    
                    image.save(image_path)
                    print(f"Image saved at {image_path}")
                    # flush gpu memory
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"Error generating image for {index}")
                    print(e)
                    continue
                



