from diffusers import StableDiffusionInpaintPipeline
from mask_image import create_mask_overlay_inference, attach_image
import torch

class Stable_diffusion_2_inpainting():
    def __init__(self, device="cpu"):
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                        "stabilityai/stable-diffusion-2-inpainting",
                        torch_dtype=torch.float16,
                     )
        if device == "cuda" and torch.cuda.is_available():
            self.pipe = self.pipe.to(device)
        else:
            self.pipe = self.pipe.to(device)

    def inference(self, image, direction, prompt):
        processed_img, mask = create_mask_overlay_inference(image, direction)
        negative_prompt = "bad architecture, unstable, poor details, blurry"
        inference = self.pipe(prompt=prompt, negative_prompt=negative_prompt, image=processed_img, mask_image=mask).images[0]
        # inference.show()
        image = attach_image(image, inference, direction)
        return image
