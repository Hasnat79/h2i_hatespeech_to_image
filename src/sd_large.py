
# Gen
import os
# Image Generation
from diffusers import StableDiffusion3Pipeline
import torch
# Image similarity
import warnings
warnings.filterwarnings('ignore')
from huggingface_hub import login

DEVICE = 'cuda' if torch.cuda.is_available else 'cpu' # device
NUM_INFERENCE_STEPS = 2 # Number of inference steps to the Diffusion Model
login("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")
model = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.float16)
# model = torch.nn.DataParallel(model)
model = model.to(DEVICE)

prompt ="A capybara holding a sign that reads Hello World"
output_image = model(prompt,
                     negative_prompts="",
                     num_inference_steps=NUM_INFERENCE_STEPS,
                     height= 32,
                     width= 32,
                     guidance_scale = 7.0).images[0]
output_image.save("output.png")
