import requests
from PIL import Image, ImageDraw
from io import BytesIO


def create_mask_overlay(image_path, direction='right'):
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