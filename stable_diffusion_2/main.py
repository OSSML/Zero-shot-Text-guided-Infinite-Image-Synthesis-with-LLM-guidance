from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image
import requests
from io import BytesIO


pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
).to("cuda")

# url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
# url2 = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/90/Flag_of_Gembloux.svg/450px-Flag_of_Gembloux.svg.png"


# response = requests.get(url)
# response2 = requests.get(url2)
# image = Image.open(BytesIO(response.content)).convert("RGB")
# mask = Image.open(BytesIO(response2.content)).convert("RGB")
image = Image.open(r"masked_dog.jpg")
mask = Image.open(r"mask.jpeg")
image = image.resize((512, 512))
mask = mask.resize((512, 512))


prompt = "Realistic image of white dog and a women sitting on a wooden bench in a field with tall grass and trees in the background"
#image and mask_image should be PIL images.
#The mask structure is white for inpainting and black for keeping as is
image = pipe(prompt=prompt, image=image, mask_image=mask).images[0]
# image = pipe(prompt="Apple on a bench")
image.save("./stable_diffusion_2.png")
