import os

import moviepy.video.io.ffmpeg_writer as ffmpeg_writer
import numpy as np
import tensorflow as tf
import Transform

from moviepy.video.io.VideoFileClip import VideoFileClip

DEVICE = '/gpu:0'
EX = Exception("Please provide a model")


def video_style_transfer(input_path, model_path, output_path, batch_s=4):

    video = VideoFileClip(input_path, audio=False)
    video_w = ffmpeg_writer.FFMPEG_VideoWriter(output_path, video.size, video.fps, codec="libx264",
                                               preset="medium", bitrate="2000k",
                                               audiofile=input_path, threads=None,
                                               ffmpeg_params=None)

    with tf.Graph().as_default(), tf.Session() as session:

        video_iter = list(video.iter_frames())
        batch_l = [video_iter[i:i + batch_s] for i in range(0, len(video_iter), batch_s)]
        while len(batch_l[-1]) < batch_s:
            batch_l[-1].append(batch_l[-1][-1])

        print("Loading model, it may take some time")
        video_wip = np.array(batch_l, dtype=np.float32)
        place_holder = tf.placeholder(tf.float32, shape=video_wip.shape[1:], name='place_holder')
        wip = Transform.net(place_holder)

        p_loader = tf.train.Saver()

        if os.path.isdir(model_path):

            model = tf.train.get_checkpoint_state(model_path)
            is_valid = model.model_checkpoint_path

            if model is not None and is_valid:
                p_loader.restore(session, is_valid)
            else:
                raise EX
        else:
            p_loader.restore(session, model_path)

        # The information about size in the video files are: 'width, height'
        # In *** the dimensions are 'height, width'
        #shape = (batch_s, video.size[1], video.size[0], 3)
        # TODO check if it's ok without shape
        for i in range(len(video_wip)):
            r_res = session.run(wip, feed_dict={place_holder: video_wip[i]})
            for r in r_res:
                video_w.write_frame(np.clip(r, 0, 255).astype(np.uint8))
            print("processed " + str(i+1) + " out of " + str(len(video_wip)) + " batches", end = '\r')

        video_w.close()