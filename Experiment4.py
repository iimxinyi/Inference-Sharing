import os
import torch
import random
import numpy as np
from diffusers import StableDiffusion3Pipeline

# Function to set seed for all random operations
def seed_everywhere(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# Local path to Stable Diffusion 3
model_directory = "/root/dataDisk/SD3MD"

# Stable Diffusion 3 pipeline
pipe = StableDiffusion3Pipeline.from_pretrained(model_directory, torch_dtype=torch.float16).to("cuda")

# Basic parameters
seed = 1
seed_everywhere(seed)
total_step = 28
common_step = 8
scale = 7.0

public_prompt = ["A tiger navigates its natural habitat, resting and preparing for the hunt in the shade."]

prompts = [
    "A tiger rests under the shade of a large tree in the jungle.",
    "A tiger slinks through the underbrush in the twilight.",
    "A tiger prowls the edge of a dense thicket, looking for prey.",
    "A tiger lounges in the shade, conserving energy for the hunt.",
    "A tiger rests under a waterfall, the cool water soothing its fur.",
]

# Get all images
for seed in range(1,6):
    for index1, prompt1 in enumerate(public_prompt):
        for index2, prompt2 in enumerate(prompts):
            for common_step in range(0,29):
                image = pipe(prompt1, num_inference_steps=total_step, guidance_scale=scale, common_step=common_step, prompt_unchanged=True).images[0]
                file_name = str(index1+1+1) + "_" + str(index2+5+5) + "_" + str(common_step) + "_original.png"
                folder_name = "/root/dataDisk/Experiment4/seed" + str(seed) + "/public_prompt2"
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                image.save(os.path.join(folder_name, file_name))

                image = pipe(prompt2, num_inference_steps=total_step, guidance_scale=scale, common_step=common_step, prompt_unchanged=False).images[0]
                file_name = str(index1+1+1) + "_" + str(index2+5+5) + "_" + str(common_step) + "_sharing.png"
                image.save(os.path.join(folder_name, file_name))
