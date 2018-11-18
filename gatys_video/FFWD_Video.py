import os

import moviepy.video.io.ffmpeg_writer as ffmpeg_writer
import numpy as np
import tensorflow as tf
import transform
from moviepy.video.io.VideoFileClip import VideoFileClip

DEVICE = '/gpu:0'
EX = Exception("Please provide a model")


def ffwd_video(input_file, output_file, p_dir, batch_s=4):
    video = VideoFileClip(input_file, audio=False)
    video_w = ffmpeg_writer.FFMPEG_VideoWriter(output_file, video.size, video.fps, codec="libx264",
                                               preset="medium", bitrate="2000k",
                                               audiofile=input_file, threads=None,
                                               ffmpeg_params=None)

    with tf.Graph().as_default(), tf.Session() as session:
        p_loader = tf.train.Saver()

        if os.path.isdir(p_dir):
            model = tf.train.get_checkpoint_state(p_dir)
            is_valid = model.model_checkpoint_path
            if model is not None and is_valid:
                p_loader.restore(session, is_valid)
            else:
                raise EX
        else:
            raise EX

        # The information about size in the video files are: 'width, height'
        # In *** the dimensions are 'height, width'
        #shape = (batch_s, video.size[1], video.size[0], 3)
        # TODO check if it's ok without shape

        video_iter = video.iter_frames()
        batch_l = [video_iter[i:i + batch_s] for i in range(0, len(video_iter), batch_s)]
        while len(batch_l[-1]) < batch_s:
            batch_l[-1].append(batch_l[-1][-1])

        video_wip = np.array(batch_l, dtype=np.float32)
        place_holder = tf.placeholder(tf.float32, shape=video_wip.shape, name='place_holder')
        wip = transform.net(place_holder)

        for ind in range(wip.shape[0]):
            r_res = session.run(wip, feed_dict={place_holder: wip[ind]})
            video_w.write_frame(np.clip(r_res, 0, 255).astype(np.uint8))
            # TODO check for the line above

        video_w.close()
