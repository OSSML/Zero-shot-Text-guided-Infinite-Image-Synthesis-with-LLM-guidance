from venv import create

from models.pretrained.stable_diffustion_2_inpainting import Stable_diffusion_2_inpainting
from mask_image import load_img

model = Stable_diffusion_2_inpainting(device="cuda")
#
image_path = r"C:\Users\Dinesh\Videos\Red Dead Redemption 2\Red Dead Redemption 2 Screenshot 2024.10.22 - 15.56.02.74.png"
# image_path = r"https://stimg.cardekho.com/images/carexteriorimages/630x420/Tata/Curvv/9578/1723033064164/front-left-side-47.jpg?"
# image_path = r"https://r4.wallpaperflare.com/wallpaper/39/346/426/digital-art-men-city-futuristic-night-hd-wallpaper-01b69d213afe95f35634472bcdf74a70.jpg"
# image, mask = create_mask_overlay_inference(image_path, 'left')
prompt = "sun behind the clouds"
prompt = "dog running behind the horse"
prompt = "grass field"
# prompt = "realistic burj kalifa to the right 4k"
# prompt = "a plane crashing the building"
# prompt = "a man to the left and a man to the right"
# prompt = "skyscraper"
#
image = load_img(image_path)

# image = model.inference(image, direction="right", prompt=prompt)
image = model.inference(image, direction="left", prompt=prompt)
# image = model.inference(image, direction="top", prompt=prompt)
image = model.inference(image, direction="right", prompt=prompt)


image.show()
# mask.show()
image.save('testing.png')