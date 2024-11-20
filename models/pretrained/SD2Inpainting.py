from diffusers import StableDiffusionInpaintPipeline
from mask_image import create_mask_overlay_inference, attach_image, create_mask_overlay_inference_v2
import torch

class SD2Inpainting():
    def __init__(self, device="cuda"):
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                        "stabilityai/stable-diffusion-2-inpainting",
                        torch_dtype=torch.float16,
                     )
        if device == "cuda" and torch.cuda.is_available():
            self.pipe = self.pipe.to(device)
        else:
            self.pipe = self.pipe.to("cpu")


    def inference(self, image, direction, prompt=None, prompt_embeds=None, similar_color=True):
        negative_prompt = "bad architecture, unstable, poor details, blurry, human, people, children"

        if similar_color:
            processed_img, mask = create_mask_overlay_inference_v2(image, direction)
            inference = self.pipe(prompt=prompt, prompt_embeds=prompt_embeds,negative_prompt=negative_prompt, image=processed_img, mask_image=mask,
                                  strength=0.999).images[0]
        else:
            processed_img, mask = create_mask_overlay_inference(image, direction)
            inference = \
            self.pipe(prompt=prompt, negative_prompt=negative_prompt, image=processed_img, mask_image=mask).images[0]

        image = attach_image(image, inference, direction)
        return image

if __name__ == "__main__":
    sd = SD2Inpainting()
