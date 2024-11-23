# Image Extension Using Stable Diffusion

This project utilizes the Stable Diffusion model to extend images based on user-provided text prompts and specified directions (e.g., left, right, top, bottom). The pipeline processes the input image and generates an extended version according to the directions specified.

## Features
- Extend images based on a text prompt.
- Specify one or more directions for the image extension: left, right, top, bottom.
- Simple and interactive user interface via Gradio.


## Usage Guide
1. Open the [Google Colab notebook](<https://colab.research.google.com/drive/1_b1qeCJO_y6f0H5_76dyjx3UAONKR2rJ>).
2. Run the setup code and run app code cells in the notebook.
3. After execution, click on the Gradio link displayed in the notebook output.
4. The Gradio link opens a live website where you can:
    1. **Upload an Image**: Provide an image you want to extend.
    2. **Enter a Prompt**: Describe the desired extension (e.g., "Lake and grass fields").
    3. **Select Directions**: Choose one or more directions (left, right, top, bottom) for the extension.
    4. **Generate**: Click the button to process and view the extended image.

## How It Works
1. **Input**: 
   - An image to be extended.
   - A text prompt describing how the extension should look.
   - The direction(s) in which the image should be extended.
   - No. of inference images to be generated.

2. **Processing**:
    - The given image is masked w.r.t. the given direction.
    - The masked image and prompt are passed to the Stable Diffusion model.
    - The model generates an extended version of the image based on the given prompt and direction(s).

3. **Output**:
   - The extended image is displayed on the live website.


## Example
- Input: 
    - Image : ![Red_Dead_Redemption_2_Screenshot_2024 10 21_-_18 44 19 46](https://github.com/user-attachments/assets/b005efa6-4eeb-46e4-88ad-159eec70c0e9)

    - Prompt : Lake and grass fields
    - Directions : `left`, `right`
- Output: ![image](https://github.com/user-attachments/assets/c638d07a-39b7-467e-9b14-977cdc7917ca)
------
- Input: 
    - Image : ![WhatsApp Image 2024-11-23 at 19 49 57_fb637476](https://github.com/user-attachments/assets/e50dd3f9-ce13-4a2d-b4c8-d666cb6384ec)


    - Prompt : Cafe and cozy outdoor
    - Directions : `top`
- Output: 

  ![image(1)](https://github.com/user-attachments/assets/8603eb30-64ce-42e8-98e0-e803b27241f7)
------

## Technical Details
- **Model**: Stable Diffusion, fine-tuned for image extension tasks.
- **Framework**: Gradio for building the interactive interface.
- **Deployment**: The system runs entirely within Google Colab for easy setup and execution.

## Project Link
[Google Colab Notebook](<INSERT_LINK_HERE>)

## License
![License](https://img.shields.io/github/license/OSSML/Zero_shot_Image_Synthesis)  
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)  

