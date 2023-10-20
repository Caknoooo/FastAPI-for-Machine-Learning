from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from PIL.Image import Image

token=None

token_path = Path("token.txt")
if token_path.exists():
    with open(token_path, "r") as f:
        token = f.read().strip()

pipe = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=token,
)

pipe.model.to("cuda")

prompt = "A painting of a cat"

image = pipe(prompt)["sample"][0]

def obtain_image(
        prompt: str,
        *, 
        seed: int | None = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
) -> Image:
    generator: None if seed is None else torch.Generator("cuda").manual_seed(seed)
    print(f"using device: {pipe.device}")
    image = pipe(
        prompt,
        seed=generator,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    ).images[0]
    return image