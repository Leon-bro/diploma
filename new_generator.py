import tensorflow as tf
import numpy as np
import preprocessing
import preprocessing as prp
import cv2
# Function to detect and decode the image based on file extension
def universal_image_decoder(image_path, channels=3):
    # Extract the file extension using TensorFlow operations
    file_extension = tf.strings.split(image_path, '.')[-1]

    # Read the file
    image_raw = tf.io.read_file(image_path)

    # Decode based on the file extension
    def decode_gif():
        image = tf.image.decode_gif(image_raw)
        return image[0]  # Extract only the first frame

    def decode_jpeg():
        return tf.image.decode_jpeg(image_raw, channels=3)

    def decode_png():
        return tf.image.decode_png(image_raw, channels=3)

    def decode_tiff():
        # Use OpenCV for TIFF decoding
        image_np = tf.io.decode_raw(image_raw, tf.uint8)
        image = tf.py_function(func=load_tiff_image_with_opencv, inp=[image_np], Tout=tf.uint8)
        return image

    # Define a placeholder tensor to handle unsupported formats
    def unsupported_format():
        # Return a tensor with zeros as a placeholder for unsupported formats
        return tf.zeros([256, 256, 3], dtype=tf.uint8)  # You can customize the shape

    # Conditional decoding based on file extension
    image = tf.case(
        [(tf.equal(file_extension, 'gif'), decode_gif),
         (tf.equal(file_extension, 'jpeg'), decode_jpeg),
         (tf.equal(file_extension, 'jpg'), decode_jpeg),
         (tf.equal(file_extension, 'png'), decode_png),
         (tf.equal(file_extension, 'tif'), decode_tiff),
         (tf.equal(file_extension, 'tiff'), decode_tiff)],
        default=unsupported_format,  # Return the placeholder tensor for unsupported formats
        exclusive=True
    )


    return image

# Helper function to load .tiff using OpenCV and handle it in TensorFlow
def load_tiff_image_with_opencv(image_raw):
    # Convert the tensor to NumPy array
    image_np = np.array(image_raw)

    # Use OpenCV to decode the .tif image (read as RGB)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)  # OpenCV reads BGR format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    return tf.convert_to_tensor(image_rgb, dtype=tf.uint8)
# Step 1: Load and preprocess image and label
def load_and_preprocess(image_path, label_path):
    # Load image
    image = tf.io.read_file(image_path)
    image = universal_image_decoder(image, channels=3)
    image = image[:, :, 1]
    tf.print(image.shape)
    tf.print(type(image))
    tf.print(tf.math.reduce_mean(image))
    # Load label (assuming it's a grayscale image, 1 channel)
    label = tf.io.read_file(label_path)
    label = universal_image_decoder(label, channels=1)
    tf.print(label.shape)
    tf.print(type(label))
    tf.print(tf.math.reduce_mean(label))
    # Contrast correction for the image and normalization
    image = tf.image.adjust_contrast(image, contrast_factor=2)
    tf.print(image.shape)
    image = tf.image.adjust_gamma(image, gamma=1.0)
    tf.print(image.shape)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize image (0-1)
    label = tf.cast(label, tf.float32)  # Keep label as float, but no normalization

    return image, label

# Step 2: Data amplification (extract patches with aligned cropping)
def extract_patches(image, label, patch_size=64, k=5):
    image_patches = []
    label_patches = []

    for _ in range(k):
        # Get image dimensions
        height, width, _ = tf.shape(image)[0], tf.shape(image)[1], tf.shape(image)[2]

        # Generate random offsets for cropping
        offset_height = tf.random.uniform([], 0, height - patch_size, dtype=tf.int32)
        offset_width = tf.random.uniform([], 0, width - patch_size, dtype=tf.int32)

        # Crop both image and label using the same offsets
        image_patch = tf.image.crop_to_bounding_box(image, offset_height, offset_width, patch_size, patch_size)
        label_patch = tf.image.crop_to_bounding_box(label, offset_height, offset_width, patch_size, patch_size)

        image_patches.append(image_patch)
        label_patches.append(label_patch)

    return tf.stack(image_patches), tf.stack(label_patches)

# Step 3: Data augmentation (apply same augmentations to both image and label)
def augment_patches(image_patches, label_patches):
    augmented_image_patches = []
    augmented_label_patches = []

    for image_patch, label_patch in zip(image_patches, label_patches):
        flip1 = tf.random.uniform([], 0, 1, dtype=tf.float32)
        flip2 = tf.random.uniform([], 0, 1, dtype=tf.float32)
        rotate = tf.random.uniform([], 0, 1, dtype=tf.float32)
        if flip1 < 0.5:
            image_patch = tf.image.flip_left_right(image_patch)
            label_patch = tf.image.flip_left_right(label_patch)
        if flip2 < 0.5:
            image_patch = tf.image.flip_up_down(image_patch)
            label_patch = tf.image.flip_up_down(label_patch)
        if rotate < 0.4:
            rotation_angle = tf.random.uniform([], -10., 10., dtype=tf.float32)
            radians = tf.constant(rotation_angle * np.pi / 180, dtype=tf.float32)  # Convert to radians
            image_patch = tf.image.rotate(image_patch, radians)
            label_patch = tf.image.rotate(label_patch, radians)
        augmented_image_patches.append(image_patch)
        augmented_label_patches.append(label_patch)

    return tf.stack(augmented_image_patches), tf.stack(augmented_label_patches)

# Building the full pipeline step-by-step for both images and labels
def preprocess_pipeline(image_path, label_path, patch_size=64, k=5):
    image, label = load_and_preprocess(image_path, label_path)
    image_patches, label_patches = extract_patches(image, label, patch_size=patch_size, k=k)
    augmented_image_patches, augmented_label_patches = augment_patches(image_patches, label_patches)

    return augmented_image_patches, augmented_label_patches