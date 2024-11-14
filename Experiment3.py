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

public_prompt = ["A dog enjoys various peaceful and playful moments in a cozy or natural setting."]

prompts = [
    "A dog sleeps peacefully on a cozy couch.",
    "A dog sniffs curiously at a flower garden.",
    "A dog pounces playfully at a butterfly in a meadow.",
    "A dog lounges lazily on a patio during a summer afternoon.",
    "A dog rests contentedly by a window with a view of the mountains.",
]

# Get all images
for seed in range(1,6):
    for index1, prompt1 in enumerate(public_prompt):
        for index2, prompt2 in enumerate(prompts):
            for common_step in range(0,29):
                image = pipe(prompt1, num_inference_steps=total_step, guidance_scale=scale, common_step=common_step, prompt_unchanged=True).images[0]
                file_name = str(index1+1) + "_" + str(index2+5) + "_" + str(common_step) + "_original.png"
                folder_name = "/root/dataDisk/Experiment3/seed" + str(seed) + "/public_prompt1"
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                image.save(os.path.join(folder_name, file_name))

                image = pipe(prompt2, num_inference_steps=total_step, guidance_scale=scale, common_step=common_step, prompt_unchanged=False).images[0]
                file_name = str(index1+1) + "_" + str(index2+5) + "_" + str(common_step) + "_sharing.png"
                image.save(os.path.join(folder_name, file_name))
