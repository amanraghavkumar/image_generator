from flask import Flask, render_template, request, send_from_directory
from diffusers import StableDiffusionPipeline
import torch
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Load Hugging Face token securely
access_token = os.getenv("HF_TOKEN")

# Load the Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=access_token
).to("cuda")

@app.route('/', methods=['GET', 'POST'])
def index():
    image_generated = False
    if request.method == 'POST':
        prompt = request.form['prompt']
        image = pipe(prompt).images[0]
        image.save("static/generated_image.png")
        image_generated = True
    return render_template('index.html', image_generated=image_generated)

@app.route('/download')
def download():
    return send_from_directory('static', 'generated_image.png', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
