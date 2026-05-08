import gradio as gr
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from train import Diffusion
from PIL import Image

model = Diffusion(inference=True)
PROJECT_ROOT = Path(__file__).parent.parent 
checkpoint = torch.load(PROJECT_ROOT/"checkpoints"/"latest.pt", map_location=model.device)
model.ema_network.load_state_dict(checkpoint['ema_network'])
model.ema_network.eval()

@torch.no_grad()
def colorize(image: Image.Image, diffusion_steps: int = 100, num_samples: int = 1):
    resized = image.resize((128, 128), Image.LANCZOS)
    gray = resized.convert("L")

    img = torch.from_numpy(np.array(gray)).float()
    img = img.div(127.5).sub(1)
    img = img[None, None, :, :].to(model.device)

    # For a batch of images for the num_samples times
    img = img.repeat(num_samples, 1, 1, 1)

    #outputs a batch of colourized images
    output = model.reverse_diffusion(img, diffusion_steps)

    results = []
    for i in range(num_samples):
        out = output[i].add(1.0).div(2.0).clamp(0, 1)
        out = (out.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        results.append(Image.fromarray(out))
    return results


with gr.Blocks(title="Image Colorizer") as demo:
    gr.Markdown("# Image Colorizer")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Image")
            diffusion_steps = gr.Slider(minimum=10, maximum=100, value=100, step=10, label="Diffusion Steps")
            num_samples = gr.Slider(minimum=1, maximum=8, value=1, step=1, label="Number of Samples")
            run_btn = gr.Button("Colorize", variant="primary")
        with gr.Column():
            output_gallery = gr.Gallery(label="Colorized Outputs", columns=2)

    run_btn.click(fn=colorize, inputs=[input_image, diffusion_steps, num_samples], outputs=output_gallery)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

