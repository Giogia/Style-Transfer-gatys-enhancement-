import tensorflow as tf
import tensorflow.contrib.eager as tfe
import Loss
import numpy as np
import os
import Image
import CNN
import moviepy.video.io.ffmpeg_writer as ffmpeg_writer
from moviepy.video.io.VideoFileClip import VideoFileClip

MEAN_PIXEL = np.array([123.68, 116.779, 103.939])
save_path = os.getcwd()

def run_style_transfer(content, style, iterations=1000, content_weight=1e0, style_weight=1e2, learning_rate=5):

    vgg = CNN.VGG19_c()

    noise = [Image.preprocess_image(Image.generate_noise_image(c)) for c in content]
    content = [Image.preprocess_image(c) for c in content]
    style = Image.preprocess_image(style)
    p = 0

    noise = [p * n + (1 - p) * c for n, c in zip(noise, content)]

    noise = [tfe.Variable(n, dtype=tf.float32) for n in noise]

    # create model
    loss_weights = content_weight, style_weight
    layers_number = vgg.content_layers_num, vgg.style_layers_num

    style_features = vgg.get_style_features(style)
    opt = tf.train.AdamOptimizer(learning_rate, beta1=0.99, epsilon=1e-1)
    # precompute style features
    content_features = [vgg.get_content_features(c) for c in content]
    gram_matrix_features = [oloss.g_matrix(feature) for feature in style_features]
    # create features
    img_features = [(cf, gram_matrix_features) for cf in content_features]
    best = [(float('inf'), None) for n in noise]
    # overall loss

    for i in range(iterations):
        print("computing iteration " + str(i+1))
        for index in range(len(content)):
            grads, loss = loss.compute_gradient(noise[index], vgg.get_output_features, img_features[index], loss_weights, layers_number)
            opt.apply_gradients([(grads, noise[index])])
            if loss < best[index][0]:
                # Update best loss and best image from total loss.
                best[index] = loss, Image.postprocess_image(noise[index].numpy())
            noise[index].assign(Image.clip_image(noise[index]))
    return [b[1] for b in best]


def video_style_transfer_gatys(video_path, style_path, output_path, batch_s=4):

    video = VideoFileClip(video_path, audio=False)
    video_w = ffmpeg_writer.FFMPEG_VideoWriter(output_path, video.size, video.fps, codec="libx264",
                                               preset="medium", bitrate="2000k",
                                               audiofile=video_path, threads=None,
                                               ffmpeg_params=None)

    style = Image.load_image(style_path)
    content = [c for c in video.iter_frames()]
    batch_l = [content[i:i + batch_s] for i in range(0, len(content), batch_s)]
    for b in batch_l:
        frames = run_style_transfer(b, style)
        for f in frames:
            video_w.write_frame(f)
    video_w.close()
