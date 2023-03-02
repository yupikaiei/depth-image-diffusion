from cog import BasePredictor, Input, Path
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionUpscalePipeline, DiffusionPipeline, StableDiffusionDepth2ImgPipeline, DPMSolverMultistepScheduler
from PIL import Image
import random

class Predictor(BasePredictor):
    def setup(self):
        # Load the model into memory to make multiple predictions
        self.pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-depth",
            revision="fp16",
            torch_dtype=torch.float16,
            # scheduler=scheduler
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_attention_slicing()
        self.pipe.to("cuda")
        return self.pipe

    def predict(self, image: Input(Image, extensions=["png", "jpg"]), prompt: str = Input(default=""), num_images: int = 1, negative_prompt: Input = "", num_steps: int = 25, guidance_scale: float = 9.0, seed: int = 0):
        # Make a prediction using the model
        if seed == 0:
            seed = random.randint(0, 2147483647)

        generator = torch.Generator('cuda').manual_seed(seed)

        return self.pipe(
            prompt=prompt,
            image=image,
            num_images_per_prompt=num_images,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )