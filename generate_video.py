import os
from PIL import Image, ImageDraw, ImageFont
import imageio

# Settings
input_dir = './val_output_tensorf/'         # Folder containing your images
output_path = 'output_tensorf.gif'  # Can also be 'output.mp4' if using ffmpeg writer
font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'  # Adjust as needed
font_size = 24

# Load font
font = ImageFont.truetype(font_path, font_size)

# Collect and sort PNGs
files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])

frames = []
for file in files:
    path = os.path.join(input_dir, file)
    image = Image.open(path).convert("RGB")

    draw = ImageDraw.Draw(image)

    # Extract epoch from filename, e.g., "epoch_5.png" → 5
    epoch = os.path.splitext(file)[0].split('_')[-1]
    text = f"epoch: {epoch}"
    draw.text((10, 10), text, font=font, fill=(255, 255, 255))

    frames.append(image)

# Save as GIF (or mp4 with imageio-ffmpeg)
frames[0].save(
    output_path,
    save_all=True,
    append_images=frames[1:],
    duration=100,  # ms per frame
    loop=0
)
