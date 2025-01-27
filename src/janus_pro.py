import os
import sys
sys.path.append("/home/grads/h/hasnat.md.abdullah/h2i_hatespeech_to_image/")
sys.path.append("/scratch/user/hasnat.md.abdullah/h2i_hatespeech_to_image/")

import torch
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import diffusers
import transformers
import accelerate
from transformers import AutoModelForCausalLM
from src.models import MultiModalityCausalLM, VLChatProcessor



print("Torch Version",torch.__version__)
print("Diffusers Version",diffusers.__version__)
print("Transformers Version",transformers.__version__)
print("Accelerate Version",accelerate.__version__)

from PIL import Image
from data import super_set_hateful_gender, super_set_hateful_disability
import shutil
import random
# image generation parameters
NUMBER_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.0

@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    temperature: float = 1,
    parallel_size: int = 16,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
):
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size*2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
        hidden_states = outputs.last_hidden_state
        
        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        
        logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)


    dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    random_image_index = random.randint(0, parallel_size - 1)
    selected_image = visual_img[random_image_index]
    return selected_image
    PIL.Image.fromarray(selected_image).save(selected_image_path)
    # for i in range(parallel_size):
    #     save_path = os.path.join('generated_samples', "img_{}.jpg".format(i))
    #     PIL.Image.fromarray(visual_img[i]).save(save_path)

def load_model (model_path):
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path, cache_dir = "./cache")
    tokenizer = vl_chat_processor.tokenizer
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, cache_dir = "./cache"
)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    return vl_gpt, vl_chat_processor

def process_prompt(prompt): 
    conversation = [
    {
        "role": "<|User|>",
        "content": f"{prompt}",
    },
    {"role": "<|Assistant|>", "content": ""},
]

    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    prompt = sft_format + vl_chat_processor.image_start_tag
    return prompt

if __name__ == "__main__":

    # Load the model
    model_id = "deepseek-ai/Janus-Pro-7B"
    vl_gpt, vl_chat_processor = load_model(model_id)


    
    output_dir = "/scratch/user/hasnat.md.abdullah/h2i_hatespeech_to_image/data/generated_images/janus_pro/hateful"
    filter_keywords = ['gender']
    # filter_keywords = ['disability']
    #-----------------#
    data_frames = [super_set_hateful_gender]
    # data_frames = [super_set_hateful_disability]
    # verify
    print(data_frames[0].shape)
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
            print(f"Prompt: {prompt}")
            # creates a directory for each index
            dir_path = os.path.join(filter_dir, f"{index}")
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            # check if the directory already has 5 images, if yes then skip
            if len([name for name in os.listdir(dir_path) if name.endswith('.png')]) == 5:
                print(f"Directory {dir_path} already has 5 images, skipping...")
                continue
            # generate n_pass images for each prompt and save them in the directory: img_i.png
            # images = generate (vl_gpt, vl_chat_processor, prompt)
            for i in range(n_pass):

                try: 
                    image_path = os.path.join(dir_path, f"img_{i+1}.png")
                    if os.path.exists(image_path):
                        print(f"Image already exists at {image_path}")
                        continue
                    prompt = process_prompt(prompt)
                    # image = generate (vl_gpt, vl_chat_processor, prompt)
                    # image = generate_image_sdxl(model, prompt, NUMBER_INFERENCE_STEPS, GUIDANCE_SCALE)
                    # print("dummy image generated")
                    # image = dummy_image()
                    image = generate (vl_gpt, vl_chat_processor, prompt)

                    Image.fromarray(image).save(image_path)
                    # image.save(image_path)
                    print(f"Image saved at {image_path}")
                    # flush gpu memory
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"Error generating image for {index}")
                    print(e)
                    continue






    






