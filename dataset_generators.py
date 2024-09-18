import cv2
from PIL import Image
from preprocessing import *

# Function to preprocess images and labels
def load_data(image_path, label_path, mask_path, image_preproc:PreprocessLayer, label_preproc:PreprocessLayer):

    # Load image and label
    image = cv2.imread(image_path)[..., 1]
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
        