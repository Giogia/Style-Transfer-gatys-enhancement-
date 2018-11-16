import os
import matplotlib.pyplot as plt

from tensorflow import clip_by_value
from numpy import clip, expand_dims, squeeze, array, random
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input


def load_image(path):

    max_dim = 1024

    path = os.getcwd() + path
    img = Image.open(path)

    # resize image to max_dim
    scale = max_dim / max(img.size)

    if scale < 1:
        scaled_width = round(img.size[0] * scale)
        scaled_height = round(img.size[1] * scale)
        img = img.resize((scaled_width, scaled_height))

    img = img_to_array(img)

    return img


def save_image(path, img):

    img = Image.fromarray(clip(img, 0, 255).astype('uint8'))
    path = os.getcwd() + path
    img.save(path, 'JPEG')


def preprocess_image(img):

    img = expand_dims(img, axis=0)

    #normalize by mean = [103.939, 116.779, 123.68] and with channels BGR
    img = preprocess_input(img)

    return img


def postprocess_image(processed_img):

    img = processed_img.copy()

    # shape (1, h, w, d) to (h, w, d)
    if len(img.shape) == 4:
        img = squeeze(img, axis=0)
    if len(img.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    # Remove VGG mean
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68

    # rgb to bgr
    img = img[:, :, ::-1]

    #cast to values within (-255,255)
    img = clip(img, 0, 255).astype('uint8')

    return img


def clip_image(img):

    norm_means = array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means

    img = clip_by_value(img, min_vals, max_vals)

    return img


def show_image(img, title=None):

    # Normalize for display
    out = img.astype('uint8')

    # Remove the batch dimension
    if len(img.shape) == 4:
        out = squeeze(out, axis=0)

    if title is not None:
        plt.title(title)

    plt.imshow(out)


def generate_noise_image(img):

    img = random.uniform(-20,20,img.shape).astype('uint8')

    return img


def show_content_style(content_path, style_path):

    plt.figure(figsize=(10,10))

    content = load_image(content_path)
    style = load_image(style_path)

    plt.subplot(1, 2, 1)
    show_image(content, 'Content')

    plt.subplot(1, 2, 2)
    show_image(style, 'Style')

    plt.show()