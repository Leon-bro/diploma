from abc import ABC, abstractmethod
import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


# Step 1: Abstract handler class
class Preprocessor(ABC):
    def __init__(self):
        self.next_handler = None

    def __call__(self, handler):
        handler.next_handler = self

    @abstractmethod
    def process(self, image):
        if self.next_handler:
            return self.next_handler.process(image)
        return image


class PreprocessLayer(Preprocessor):
    def __call__(self, handler):
        super().__call__(handler)
        return self


class CropBlackBox(PreprocessLayer):
    def __init__(self):
        super().__init__()

    def process(self, image):
        assert len(image.shape) == 2
        assert image is not None, "file could not be read, check with os.path.exists()"

        def crop_black_box(image):
            """
            Crops the black border from the image.
            
            Parameters:
            image (numpy.ndarray): Input image with a possible black border.
            
            Returns:
            cropped_image (numpy.ndarray): Image with black border cropped out.
            """
            # Convert to grayscale if the image is colored (RGB or BGR)
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image  # Already grayscale

            # Threshold the image to binary: pixels below 10 intensity are black
            _, thresh = cv2.threshold(gray, gray.max()*0.06, gray.max(), cv2.THRESH_BINARY)

            # Find the coordinates of the non-black pixels
            coords = np.column_stack(np.where(thresh > 0))

            # Get the bounding box of the non-black area
            if coords.size == 0:
                # If the image is entirely black, return the original image
                return image

            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)

            # Crop the image using the bounding box
            cropped_image = image[x_min:x_max + 1, y_min:y_max + 1]

            return cropped_image
        cropped_image = crop_black_box(image)
        return super().process(cropped_image)


class CLAHE(PreprocessLayer):
    def __init__(self, clipLimit=2.0, tileGridSize=(8, 8)):
        super().__init__()
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize

    def process(self, image):
        assert len(image.shape) == 2
        assert image is not None, "file could not be read, check with os.path.exists()"
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        preprocessed = clahe.apply(image)
        return super().process(preprocessed)


# Step 2: Concrete handler for resizing
class ResizePreprocessor(PreprocessLayer):
    def __init__(self, width=512, height=512):
        super().__init__()
        self.width = width
        self.height = height

    def process(self, image):
        resized_image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        logging.debug("Resized image")
        return super().process(resized_image)


# Step 2: Concrete handler for converting to grayscale
class GrayscalePreprocessor(PreprocessLayer):
    def process(self, image):
        assert len(image.shape) == 3, "image already has one chanel"
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logging.debug("Converted to grayscale")
        return super().process(gray_image)


# Step 2: Concrete handler for normalization
class NormalizePreprocessor(PreprocessLayer):
    def process(self, image):
        norm_image = image
        if image.max() > 1.0:
            norm_image = (image - image.min())/(image.max()-image.min())#image / 255.0  # Normalizing pixel values between 0 and 1
            logging.debug("Normalized image")
        return super().process(norm_image)


class MedianBlurPreprocessor(PreprocessLayer):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size

    def process(self, image):
        final = cv2.medianBlur(image, self.kernel_size)
        logging.debug("Blured image")
        return super().process(final)


# Step 2: Concrete handler for adaptive gamma correction
class GammaCorrectionPreprocessor(PreprocessLayer):
    def __init__(self, gamma=None):
        super().__init__()
        self.gamma = gamma

    def adaptive_gamma(self, image):
        # Calculate the mean brightness of the image
        mean_brightness = np.mean(image) / 255.0

        # Dynamically adjust gamma based on brightness (low brightness -> higher gamma)
        if mean_brightness > 0.5:
            gamma = 0.7  # lower gamma for brighter images
        else:
            gamma = 1.5  # higher gamma for darker images

        return gamma

    def apply_gamma_correction(self, image, gamma):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def process(self, image):
        if self.gamma is None:
            self.gamma = self.adaptive_gamma(image)

        corrected_image = self.apply_gamma_correction(image, self.gamma)
        logging.debug(f"Applied gamma correction with gamma: {self.gamma}")
        return super().process(corrected_image)


# Step 2: Concrete handler for multi-scale morphological transformation
class MultiScaleMorphologicalPreprocessor(PreprocessLayer):
    def __init__(self, operation='open', kernel_sizes=[3, 5, 7], kernel_shape=cv2.MORPH_RECT, iterations=1):
        super().__init__()
        self.operation = operation.lower()
        self.kernel_sizes = kernel_sizes
        self.kernel_shape = kernel_shape
        self.iterations = iterations

    def process(self, image):
        # Ensure image is in uint8 format
        if image.dtype != np.uint8:
            image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
            logging.debug("Converted image to uint8 for morphological operations")
        else:
            image_uint8 = image

        transformed_images = []
        for size in self.kernel_sizes:
            kernel = cv2.getStructuringElement(self.kernel_shape, (size, size))
            if self.operation == 'dilate':
                morphed = cv2.dilate(image_uint8, kernel, iterations=self.iterations)
            elif self.operation == 'erode':
                morphed = cv2.erode(image_uint8, kernel, iterations=self.iterations)
            elif self.operation == 'open':
                morphed = cv2.morphologyEx(image_uint8, cv2.MORPH_OPEN, kernel, iterations=self.iterations)
            elif self.operation == 'close':
                morphed = cv2.morphologyEx(image_uint8, cv2.MORPH_CLOSE, kernel, iterations=self.iterations)
            else:
                raise ValueError(f"Unsupported morphological operation: {self.operation}")
            logging.debug(f"Applied {self.operation} with kernel size {size}")
            transformed_images.append(morphed)

        # Combine the transformed images
        combined_image = transformed_images[0]
        for img in transformed_images[1:]:
            if self.operation in ['dilate', 'open', 'close']:
                combined_image = cv2.max(combined_image, img)
            elif self.operation == 'erode':
                combined_image = cv2.min(combined_image, img)

        logging.debug("Combined multi-scale morphological transformations")
        return super().process(combined_image)
