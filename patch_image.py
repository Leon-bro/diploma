import numpy as np
from PIL import Image

def split_image_into_patches(image, patch_size):
    """
    Splits an image into patches of size patch_size x patch_size.

    Args:
    image (PIL.Image): The input image to split.
    patch_size (int): The size of each patch (n x n).

    Returns:
    patches (list): A list of image patches.
    original_size (tuple): The original size of the image before padding.
    padded_image (PIL.Image): The padded image.
    """
    # Convert image to NumPy array
    img_array = np.array(image)
    original_height, original_width = img_array.shape[:2]

    # Calculate padding needed
    pad_height = (patch_size - original_height % patch_size) % patch_size
    pad_width = (patch_size - original_width % patch_size) % patch_size

    # Pad the image with black (zero) pixels if necessary
    padded_image = np.pad(
        img_array,
        ((0, pad_height), (0, pad_width), (0, 0)),
        mode='constant',
        constant_values=0
    )

    # Get padded image size
    padded_height, padded_width = padded_image.shape[:2]

    # Split into patches
    patches = []
    for i in range(0, padded_height, patch_size):
        for j in range(0, padded_width, patch_size):
            patch = padded_image[i:i+patch_size, j:j+patch_size]
            patches.append(patch)

    # Return patches and original size (before padding)
    return patches, (original_height, original_width)

def merge_patches_into_image(patches, original_size, patch_size):
    """
    Merges patches back into the original image size.

    Args:
    patches (list): A list of image patches.
    original_size (tuple): The original size of the image before padding.
    patch_size (int): The size of each patch (n x n).

    Returns:
    PIL.Image: The reassembled image.
    """
    original_height, original_width = original_size

    # Calculate padded image size
    padded_height = original_height + (patch_size - original_height % patch_size) % patch_size
    padded_width = original_width + (patch_size - original_width % patch_size) % patch_size

    # Reconstruct the image from patches
    reconstructed_image = np.zeros((padded_height, padded_width, 1), dtype=np.float32)

    patch_index = 0
    for i in range(0, padded_height, patch_size):
        for j in range(0, padded_width, patch_size):
            reconstructed_image[i:i+patch_size, j:j+patch_size] = patches[patch_index]
            patch_index += 1

    # Remove the padding and return the image
    final_image = reconstructed_image[:original_height, :original_width]
    return final_image
