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
seed = 2024
seed_everywhere(seed)
total_step = 28
offloading_step = 8
scale = 7.0
prompts = [
    "A cat is sleeping peacefully on a sunlit window sill.",
    "A cat is playing with a ball of yarn in a cozy living room.",
    "A cat is sitting on a bookshelf surrounded by books.",
    "A cat is grooming itself on a soft blanket.",
    "A cat is lying on a windowsill, eyes closed, in a quiet library.",
    "A dog sleeps peacefully on a cozy couch.",
    "A dog sniffs curiously at a flower garden.",
    "A dog pounces playfully at a butterfly in a meadow.",
    "A dog lounges lazily on a patio during a summer afternoon.",
    "A dog rests contentedly by a window with a view of the mountains.",
    "A tiger rests under the shade of a large tree in the jungle.",
    "A tiger slinks through the underbrush in the twilight.",
    "A tiger prowls the edge of a dense thicket, looking for prey.",
    "A tiger lounges in the shade, conserving energy for the hunt.",
    "A tiger rests under a waterfall, the cool water soothing its fur.",
    "A panda sleeps peacefully in a sunny spot in the forest.",
    "A panda munches on bamboo while sitting in a tree hollow.",
    "A panda walks along a path in a bamboo forest at dusk.",
    "A panda munches on bamboo while sitting on a log.",
    "A panda walks along a path in a bamboo forest at sunrise."
]

# Demo
# prompt1 = prompts[0]
# image = pipe(prompt1, num_inference_steps=total_step, guidance_scale=scale, offloading_step=offloading_step, prompt_unchanged=True).images[0]
# image.save("first_image.png")

# prompt2 = prompts[1]
# image = pipe(prompt2, num_inference_steps=total_step, guidance_scale=scale, offloading_step=offloading_step, prompt_unchanged=False).images[0]
# image.save("second_image.png")

# Get all images
for index1, prompt1 in enumerate(prompts):
    for index2, prompt2 in enumerate(prompts):
        for offloading_step in range(0,29):
            image = pipe(prompt1, num_inference_steps=total_step, guidance_scale=scale, offloading_step=offloading_step, prompt_unchanged=True).images[0]
            file_name = str(index1) + "_" + str(index2) + "_" + str(offloading_step) + "_original.png"
            folder_name = "/root/dataDisk/" + str(index1)
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            image.save(os.path.join(folder_name, file_name))

            image = pipe(prompt2, num_inference_steps=total_step, guidance_scale=scale, offloading_step=offloading_step, prompt_unchanged=False).images[0]
            file_name = str(index1) + "_" + str(index2) + "_" + str(offloading_step) + "_sharing.png"
            image.save(os.path.join(folder_name, file_name))

