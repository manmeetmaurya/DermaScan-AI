import numpy as np
from PIL import Image

IMG_SIZE = (224, 224)

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess a PIL Image for model inference.
    - Resize to 224x224
    - Convert to RGB
    - Normalize pixel values to [0, 1]
    - Add batch dimension
    """
    # Convert to RGB in case it's RGBA or grayscale
    image = image.convert("RGB")

    # Resize
    image = image.resize(IMG_SIZE)

    # Convert to numpy array and normalize
    img_array = np.array(image) / 255.0

    # Add batch dimension: (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array.astype(np.float32)


def validate_image(image: Image.Image) -> tuple[bool, str]:
    """
    Validates that the uploaded image is suitable for skin disease detection.
    Returns (is_valid, message).
    """
    try:
        width, height = image.size

        if width < 50 or height < 50:
            return False, "Image is too small. Please upload an image of at least 50x50 pixels."

        if image.mode not in ["RGB", "RGBA", "L"]:
            return False, f"Unsupported image mode: {image.mode}. Please upload a standard image."

        return True, "Image is valid."

    except Exception as e:
        return False, f"Error validating image: {str(e)}"
