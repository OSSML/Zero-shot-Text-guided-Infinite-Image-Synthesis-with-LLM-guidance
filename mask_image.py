import requests
from PIL import Image, ImageDraw
from io import BytesIO


def create_mask_overlay_test(image_path, direction='right'):
    """resize the image and mask the image in the given direction. Takes right side as the default direction"""

    if image_path.startswith("https"):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert("RGBA").resize((512,512))
    else:
        image = Image.open(image_path).convert("RGBA").resize((512, 512))

    w, h = image.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0)) 
    draw = ImageDraw.Draw(overlay)

    fill = (0, 0, 0, 255)

    if direction == 'top':
        draw.rectangle([0, 0, w, h // 2], fill=fill)
    elif direction == 'bottom':
        draw.rectangle([0, h // 2, w, h], fill=fill)
    elif direction == 'left':
        draw.rectangle([0, 0, w // 2, h], fill=fill)
    elif direction == 'right':
        draw.rectangle([w // 2, 0, w, h], fill=fill)

    masked_image = Image.alpha_composite(image, overlay)

    return masked_image

def load_img(image_path):
    if image_path.startswith("https"):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")

    return image

def resize_img(image: Image, direction: str):
    w, h = image.size
    if direction in ['top', 'bottom']:
        # Resize the image to a width of 512 pixels, adjusting height to maintain aspect ratio
        base_width = 512
        wpercent = (base_width / float(w))
        hsize = int((float(image.size[1]) * float(wpercent)))
        # print("resize height: ", hsize)
        # ┌────────────────────────┐
        # │ Resize Width = 512     │  ← Set the target width to 512 pixels
        # │ Calculate Height       │  ← Adjust height proportionally
        # └────────────────────────┘

        image = image.resize((base_width, hsize), Image.Resampling.LANCZOS)
        # print("after resizing size: ", image.size)
        # ┌───────────────────────────┐
        # │ Resize Image             │
        # │ New Size: (512, hsize)   │  ← Resulting size after resizing
        # └───────────────────────────┘
    elif direction in ['left', 'right']:
        # Resize the image to a height of 512 pixels, adjusting width to maintain aspect ratio
        base_height = 512
        hpercent = (base_height / float(h))
        wsize = int((float(image.size[0]) * float(hpercent)))
        # print("resize width: ", wsize)
        # ┌───────────────────────────┐
        # │ Resize Height = 512       │  ← Set the target height to 512 pixels
        # │ Calculate Width           │  ← Adjust width proportionally
        # └───────────────────────────┘

        image = image.resize((wsize, base_height), Image.Resampling.LANCZOS)
        # print("after resizing size: ", image.size)
        # ┌──────────────────────────┐
        # │ Resize Image             │
        # │ New Size: (wsize, 512)   │  ← Resulting size after resizing
        # └──────────────────────────┘
    else:
        raise ValueError("direction must be in ['right', 'left', 'top', 'bottom']")

    return image

def crop_image_dir(image: Image, direction: str):
    base_width, base_height = 512, 512
    wsize, hsize = image.size
    image_size = 256
    # Crop image based on direction
    if direction == 'top':
        image = image.crop((0, 0, base_width, min(image_size, hsize)))
        # ┌─────────────────────┐
        # │ Crop from the top   │  ← Keep only the top 512 pixels in height
        # └─────────────────────┘
    elif direction == 'bottom':
        image = image.crop((0, max(hsize - image_size, 0), base_width, hsize))
        # ┌──────────────────────┐
        # │ Crop from the bottom │  ← Keep only the bottom 512 pixels in height
        # └──────────────────────┘
    elif direction == 'left':
        image = image.crop((0, 0, min(image_size, wsize), base_height))
        # ┌────────────────────┐
        # │ Crop from the left │  ← Keep only the left 512 pixels in width
        # └────────────────────┘
    elif direction == 'right':
        image = image.crop((max(wsize - image_size, 0), 0, wsize, base_height))
        # ┌─────────────────────┐
        # │ Crop from the right │  ← Keep only the right 512 pixels in width
        # └─────────────────────┘
    return image

def create_mask_overlay_inference(image, direction='right'):
    """Resize the image to 512x512, then create a masked extension in the specified direction, resulting in a 758x512 image."""
    if direction not in ['right', 'left', 'top', 'bottom']:
        raise ValueError("direction must be in ['right', 'left', 'top', 'bottom']")

    # image.show()
    # print("Original size: ", image.size)
    # ┌──────────────┐
    # │   w, h       │  ← Width and height of the image
    # └──────────────┘

    image = resize_img(image, direction)
    image = crop_image_dir(image, direction)

    # print("after cropping size: ", image.size)
    # image.show(title=direction)

    mask_size = 256

    if direction in ['left', 'right']:
        extended_size = (min(758, image.size[0]+mask_size), 512)
    else:
        extended_size = (512, min(758, image.size[1]+mask_size))

    # print(f"{extended_size=}")
    extended_image = Image.new("RGB", extended_size, (0, 0, 0, 0))

    if direction == 'right':
        extended_image.paste(image, (0, 0))
        mask_position = (image.size[0], 0, image.size[0] + mask_size, image.size[1])
    elif direction == 'left':
        extended_image.paste(image, (mask_size, 0))
        mask_position = (0, 0, mask_size, image.size[1])
    elif direction == 'top':
        extended_image.paste(image, (0, mask_size))
        mask_position = (0, 0, image.size[0], mask_size)
    elif direction == 'bottom':
        extended_image.paste(image, (0, 0))
        mask_position = (0, image.size[1], image.size[0], image.size[1]+mask_size)
    else:
        raise ValueError("direction must be in ['right', 'left', 'top', 'bottom']")

    draw = ImageDraw.Draw(extended_image)
    fill = (0, 0, 0, 255)
    draw.rectangle(mask_position, fill=fill)
    # extended_image.show()

    mask = Image.new("RGB", extended_image.size)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rectangle(mask_position, fill="white")
    # mask.show()

    return extended_image, mask

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def attach_image(original, inference: Image, direction: str):
    original = resize_img(original, direction)
    gen_size = 256

    if direction == 'right':
        mask_position = (inference.size[0] - gen_size, 0, inference.size[0], inference.size[1])
        inference = inference.crop(mask_position)
        result = get_concat_h(original, inference)
    elif direction == 'left':
        mask_position = (0, 0, gen_size, inference.size[1])
        inference = inference.crop(mask_position)
        result = get_concat_h(inference, original)
    elif direction == 'top':
        mask_position = (0, 0, inference.size[0], gen_size)
        inference = inference.crop(mask_position)
        result = get_concat_v(inference, original)
    elif direction == 'bottom':
        mask_position = (0, inference.size[1] - gen_size, inference.size[0], inference.size[1])
        inference = inference.crop(mask_position)
        result = get_concat_v(original, inference)
    else:
        raise ValueError("direction must be in ['right', 'left', 'top', 'bottom']")

    # inference.show()
    # result.show()
    return result


if __name__ == "__main__":
    # for dire in ['right', 'left', 'top', 'bottom']:
    # for dire in ['top']:
    #     print(f"\ndirection: {dire}")
    #     # image = create_mask_overlay_inference(r"https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg", dire)
    #     image, mask = create_mask_overlay_inference("https://images.ctfassets.net/hrltx12pl8hq/28ECAQiPJZ78hxatLTa7Ts/2f695d869736ae3b0de3e56ceaca3958/free-nature-images.jpg", dire)
    #     image.show(title=str(dire))
    #     mask.show(title=str(dire))
    image_path = 'testing.png'
    original = load_img(r"C:\Users\Dinesh\Videos\Red Dead Redemption 2\Red Dead Redemption 2 Screenshot 2024.10.22 - 15.56.02.74.png")
    inference = load_img('testing.png')
    attach_image(original, inference, 'left')
    # image.show()