import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input



def load_img(path):

    max_dim = 512

    path = os.getcwd() + path
    img = Image.open(path)

    #resize image to max_dim
    scale = max_dim/max(img.size)
    scaled_width = round(img.size[0]*scale)
    scaled_height = round(img.size[1]*scale)
    img = img.resize((scaled_width, scaled_height))

    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    return img



def load_preprocess_img(path):

    img = load_img(path)

    # adequate image to the format the model requires
    img = preprocess_input(img)

    return img



def postprocess_img(processed_img):

    img = processed_img.copy()

    # shape (1, h, w, d) to (h, w, d)
    if len(img.shape) == 4:
        img = np.squeeze(img, axis=0)
    if len(img.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    # Remove the batch dimension
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68

    # rgb to bgr
    img = img[:, :, ::-1]

    img = np.clip(img, 0, 255).astype('uint8')

    return img



def show_img(img, title=None):

    # Normalize for display
    out = img.astype('uint8')

    # Remove the batch dimension
    if len(img.shape) == 4:
        out = np.squeeze(out, axis=0)

    if title is not None:
        plt.title(title)

    plt.imshow(out)



def show_content_style(content_path, style_path):

    plt.figure(figsize=(10,10))

    content = load_img(content_path)
    style = load_img(style_path)

    plt.subplot(1, 2, 1)
    show_img(content, 'Content')

    plt.subplot(1, 2, 2)
    show_img(style, 'Style')

    plt.show()



if __name__ == "__main__":

    show_content_style('/Images/Picture1.jpg','/Images/Picture2.jpg')
