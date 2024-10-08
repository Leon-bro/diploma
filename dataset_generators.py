import cv2
from PIL import Image
from preprocessing import *
import tensorflow as tf
import random

# Function to preprocess images and labels
def load_data(image_path, label_path, mask_path, image_preproc:PreprocessLayer, label_preproc:PreprocessLayer):

    # Load image and label
    image = cv2.imread(image_path)
    image = image[...,1]
    try:
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if label is None or mask is None:
            raise FileNotFoundError
    except FileNotFoundError:
        lb = Image.open(label_path)
        lb.seek(lb.tell())
        label = np.array(lb)
        mask = Image.open(mask_path)
        mask.seek(mask.tell())
        mask = np.array(mask).astype(np.uint8)
    image = image_preproc.process(image)[:, :, np.newaxis].astype('float32')
    label = label_preproc.process(label)[:, :, np.newaxis].astype('float32')
    mask = label_preproc.process(mask)[:, :, np.newaxis].astype('float32')
    image *= mask
    label *= mask
    return image, label

def generator(image_paths, label_paths, mask_paths, image_preproc:PreprocessLayer, label_preproc:PreprocessLayer):
    for ip, lp, mp in zip(image_paths, label_paths, mask_paths):
        image, label = load_data(ip, lp, mp, image_preproc, label_preproc)
        yield image, label

def generator_with_amplification(image_paths, label_paths, mask_paths,image_preproc:PreprocessLayer, label_preproc:PreprocessLayer, data_size=1000, patch_size=48):
    data_extractor = generator(image_paths, label_paths, mask_paths, image_preproc, label_preproc)
    data_per_image = int(data_size/len(image_paths))
    for image, label in data_extractor:
        img_height, img_width, _ = image.shape
        for _ in range(data_per_image):
            # Choose a random center point (y, x) for the patch
            center_y = np.random.randint(patch_size // 2, img_height - patch_size // 2)
            center_x = np.random.randint(patch_size // 2, img_width - patch_size // 2)

            # Calculate the top-left corner of the 48x48 patch
            top_left_y = center_y - patch_size // 2
            top_left_x = center_x - patch_size // 2

            # Extract the 48x48 patch from both image and label
            img_patch = image[top_left_y:top_left_y + patch_size, top_left_x:top_left_x + patch_size, :]
            lbl_patch = label[top_left_y:top_left_y + patch_size, top_left_x:top_left_x + patch_size, :]

            yield img_patch, lbl_patch  # Yield each patch one at a time

def rotate_image(image, angle):
    """Rotate the image by a specified angle while keeping the original size."""
    # Get the image dimensions
    (h, w) = image.shape[:2]

    # Calculate the center of the image
    center = (w // 2, h // 2)

    # Generate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)  # 1.0 means no scaling

    # Perform the rotation and return the image
    rotated = cv2.warpAffine(image, M, (w, h))  # Use .numpy() to convert tensor to numpy array
    return tf.convert_to_tensor(rotated)  # Convert back to tensor

def horizontal_flip(image):
    """Perform horizontal flip on image and label."""
    image_flipped = cv2.flip(image, 1)  # 1 for horizontal flip
    return image_flipped

def vertical_flip(image):
    """Perform vertical flip on image and label."""
    image_flipped = cv2.flip(image, 0)  # 0 for vertical flip
    return image_flipped

def generator_with_augmentation(image_paths, label_paths, mask_paths,image_preproc:PreprocessLayer, label_preproc:PreprocessLayer, data_size=1000, patch_size=48):
    data_extractor = generator_with_amplification(image_paths, label_paths, mask_paths, image_preproc, label_preproc, data_size=data_size, patch_size=patch_size)
    for image, label in data_extractor:
        # Apply augmentations
        # Random horizontal flip with 50% probability
        if tf.random.uniform(()) < 0.5:
            image = horizontal_flip(image)
            label = horizontal_flip(label)

        # Random vertical flip with 50% probability
        if tf.random.uniform(()) < 0.5:
            image = vertical_flip(image)
            label = vertical_flip(label)

        # Random rotation with 40% probability, between -10 to 10 degrees
        if tf.random.uniform(()) < 0.4:
            degrees = random.uniform(-10, 10)
            image = rotate_image(image, degrees)  # Apply a random rotation to the image
            label = rotate_image(label, degrees)  # Apply the same rotation to the label
        if len(image.shape)==2:
            image = image[:, :, np.newaxis]
        if len(label.shape)==2:
            label = label[:, :, np.newaxis]
        if tf.reduce_max(image)>1.:
            image = image / 255.
        if tf.reduce_max(label)>1.:
            label = label / 255.
        yield image, label  # Yield each patch one at a time