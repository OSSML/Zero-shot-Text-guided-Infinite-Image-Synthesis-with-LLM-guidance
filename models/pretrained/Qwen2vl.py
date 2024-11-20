from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
import requests
from io import BytesIO

from LLM.llm_test import output_text


class Qwen2vl():
    def __init__(self, device="cuda"):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.bfloat16
                     )
        if device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to(device)
        else:
            self.model = self.model.to("cpu")
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels,
                                                  max_pixels=max_pixels)

    def load_img(self, image_path):
        if image_path.startswith("https"):
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_path).convert("RGB")

        return image

    def inference(self, inputs):
        inputs = inputs.to("cuda")
        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=64)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text


    def caption_image(self, image):
        prompt = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text",
                     "text": "Create a short sentence describing the image."},
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(prompt)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        output_text = self.inference(inputs)

        return output_text


if __name__ == "__main__":
    model = Qwen2vl()
    image_path = r"C:\Users\Dinesh\Pictures\parthav.jpg"
    image = model.load_img(image_path)
    # ['The image shows a person sitting on a rocky surface, facing a misty landscape. The person is wearing a dark-colored jacket and sunglasses. The background features a hazy view of mountains and low-lying clouds, creating a serene and tranquil atmosphere. The scene is likely taken during sunrise or sunset, given the soft lighting']
    model.caption_image(image)