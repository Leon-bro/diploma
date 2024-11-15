import numpy as np
from PIL import Image
import cv2
def read_ppm(file_path):
    with open(file_path, 'rb') as f:
        # Read header
        magic_number = f.readline().decode().strip()  # P3 or P6
        if magic_number not in ['P3', 'P6']:
            raise ValueError("Unsupported PPM format")

        # Skip comment lines
        line = f.readline().decode().strip()
        while line.startswith('#'):
            line = f.readline().decode().strip()

        # Read width, height, and max color value
        width, height = map(int, line.split())
        max_val = int(f.readline().decode().strip())

        # Read pixel data
        if magic_number == 'P3':
            pixels = []
            for _ in range(width * height):
                r = int(f.readline().strip())
                g = int(f.readline().strip())
                b = int(f.readline().strip())
                pixels.append((r, g, b))
        elif magic_number == 'P6':
            pixels = []
            for _ in range(width * height):
                rgb = f.read(3)
                r, g, b = rgb
                pixels.append((r, g, b))
        # Convert the pixel data to a NumPy array
        pixels_np = np.array(pixels, dtype=np.uint8)

        # Reshape it to (height, width, 3)
        image = pixels_np.reshape((height, width, 3))
        return width, height, max_val, image
    
def read_static_image(file_path):
    im = cv2.imread(file_path)
    return im
def read_gif(file_path):
    im = Image.open(file_path)
    im.seek(im.tell())
    im = np.array(im)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    return im

def read_image(file_path):
    im = None
    if file_path.endswith('.gif'):
        im = read_gif(file_path)
    else:
        im = read_static_image(file_path)
    assert im is not None, "File cannot be read"
    return im