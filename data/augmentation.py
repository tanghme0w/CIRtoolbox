import random
from PIL import Image
import numpy as np


def cutout(image, mask_size=50, mask_color=(0, 0, 0), **kwargs):
    """
    Performs the cutout operation on a PIL image.

    Args:
        image (PIL.Image): The input image to apply cutout on.
        mask_size (int): The size of the square mask to apply.
        mask_color (tuple): The color of the mask to apply (RGB values). Default is black (0, 0, 0).

    Returns:
        PIL.Image: The image after applying the cutout operation.
    """
    # Convert the image to numpy array
    np_image = np.array(image)
    
    # Get image dimensions (height, width, channels)
    # Note there may be grayscale images
    h, w = np_image.shape[0:2] 

    # Randomly choose the top-left corner of the mask
    y = random.randint(0, h - mask_size)
    x = random.randint(0, w - mask_size)

    # handle grayscale image
    if len(np_image.shape) == 2:
        # Convert the tuple mask color to a single grayscale value
        mask_color = int(np.mean(mask_color))

    # Apply the cutout mask: a square block of specified color
    np_image[y:y+mask_size, x:x+mask_size] = mask_color

    # Convert the numpy array back to PIL image
    return Image.fromarray(np_image)


def modify_hsv(image, hgain=0.1, sgain=0.7, vgain=0.7, **kwargs):
    """
    Performs HSV modification on a PIL image.

    Args:
        image (PIL.Image): The input image to apply HSV modification.
        hgain (float): Gain for modifying hue (0.0 - 1.0).
        sgain (float): Gain for modifying saturation (0.0 - 1.0).
        vgain (float): Gain for modifying value/brightness (0.0 - 1.0).

    Returns:
        PIL.Image: The image after applying the HSV modification.
    """
    # Convert the image to HSV (Hue, Saturation, Value)
    image_hsv = image.convert('HSV')

    # Convert the PIL Image to a NumPy array for processing
    np_img_hsv = np.array(image_hsv, dtype=np.float32)  # Cast to float for manipulation

    # Separate the channels: Hue, Saturation, and Value
    h, s, v = np_img_hsv[..., 0], np_img_hsv[..., 1], np_img_hsv[..., 2]

    # Modify the Hue, Saturation, and Value channels by applying the respective gains
    h = (h + hgain * 255) % 255  # Adjust hue, wrapping around the circular hue spectrum
    s = np.clip(s * (1 + sgain), 0, 255)  # Adjust saturation with gain, clip between 0 and 255
    v = np.clip(v * (1 + vgain), 0, 255)  # Adjust value with gain, clip between 0 and 255

    # Combine the modified channels back
    np_img_hsv[..., 0], np_img_hsv[..., 1], np_img_hsv[..., 2] = h, s, v

    # Convert back to uint8 for image display and manipulation
    np_img_hsv = np_img_hsv.astype(np.uint8)

    # Convert the NumPy array back to a PIL Image in HSV mode
    image_hsv_modified = Image.fromarray(np_img_hsv, mode='HSV')

    # Convert the HSV image back to RGB for output
    return image_hsv_modified.convert('RGB')


def rotate_image(image, angle, expand=True, **kwargs):
    """
    Rotates a PIL image by the specified angle.

    Args:
        image (PIL.Image): The input image to rotate.
        angle (float): The angle to rotate the image, in degrees.
                       Positive values rotate counter-clockwise.
        expand (bool): Whether to expand the output image to fit the rotated image.
                       If False, the output image size will be the same as the input size.

    Returns:
        PIL.Image: The rotated image.
    """
    # Perform the rotation using the PIL Image rotate method
    rotated_image = image.rotate(angle, expand=expand)
    
    return rotated_image


def random_scaling(image, scale_range=(0.5, 1.5), **kwargs):
    """
    Performs random scaling on a PIL image within the given scale range.

    Args:
        image (PIL.Image): The input image to scale.
        scale_range (tuple): A tuple representing the minimum and maximum scaling factors.
                             For example, (0.5, 1.5) scales between 50% and 150% of the original size.

    Returns:
        PIL.Image: The scaled image.
    """
    # Choose a random scaling factor within the specified range
    scale_factor = random.uniform(scale_range[0], scale_range[1])

    # Get original dimensions of the image
    original_width, original_height = image.size

    # Compute new dimensions based on the random scale factor
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # Resize the image using the computed dimensions with the LANCZOS filter
    scaled_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return scaled_image


def add_gaussian_noise(image, mean=0, std=25, **kwargs):
    """
    Adds Gaussian noise to a PIL image.

    Args:
        image (PIL.Image): The input image to which Gaussian noise will be added.
        mean (float): Mean of the Gaussian distribution.
        std (float): Standard deviation of the Gaussian distribution.

    Returns:
        PIL.Image: The image with added Gaussian noise.
    """
    # Convert the image to a numpy array
    np_image = np.array(image)

    # Generate Gaussian noise
    noise = np.random.normal(mean, std, np_image.shape).astype(np.float32)

    # Add the noise to the image
    noisy_image = np_image + noise

    # Clip the values to be in the valid range [0, 255] for image data
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    # Convert back to PIL Image
    return Image.fromarray(noisy_image)


def data_augmentation(image, p, **kwargs):
    """ A random combination of all augmentation methods

    Args:
        image (PIL.Image): input image.
        p (float): probability for each method being applied.
    
    Returns:
        PIL.Image: The image after modification.
    """
    methods = [cutout, modify_hsv, rotate_image, random_scaling, add_gaussian_noise]
    for method in methods:
        if random.random() < p:
            image = method(image, **kwargs)

    return image

# Example usage
if __name__ == "__main__":
    image = Image.open("/share/home/tanghaomiao/cir/fashion-iq/images/B003WTJCXW.png")
    # image = cutout(image, mask_size=50, mask_color=(0, 0, 0))
    # image = modify_hsv(image, hgain=0.1, sgain=0.3, vgain=0.5)
    # image = rotate_image(image, angle=45, expand=True)
    # image = random_scaling(image)
    # image = add_gaussian_noise(image)
    image = data_augmentation(
        image,
        p=0.2,
        mask_size=50,
        mask_color=(0, 0, 0),
        hgain=0.1,
        sgain=0.3,
        vgain=0.5,
        angle=45,
        expand=True,
        scale_range=(0.5, 1.5),
        mean=0,
        std=25,
    )
    image.save('test.png')
