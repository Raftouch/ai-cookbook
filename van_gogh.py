from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float32
).to("cpu")

image = pipe("A painting in the style of Vincent Van Gogh, something like The Starry Night").images[0]
image.save("van_gogh_style.png")
