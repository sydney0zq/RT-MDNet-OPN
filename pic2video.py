#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 qiang.zhou <theodoruszq@gmail.com>

"""

"""

import cv2
import os

def pic2video(img_dir, video_name, fps):
    images = [img for img in os.listdir(img_dir) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(img_dir, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('M', 'P', '4', '2'), fps, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(img_dir, image)))
    cv2.destroyAllWindows()
    video.release()



if __name__ == "__main__":
    img_dir = "./dump/blackswan/1/"
    video_name = "test.avi"
    from opts.track_opts import opts
    from data_prov import DAVISRegionDataset
    davis = DAVISRegionDataset(opts, data_preprocess=False)

    for _, _, seq_name, label_id in davis:
        img_dir = "./dump/{}/{}".format(seq_name, label_id)
        video_name = "videos/{}_{}.avi".format(seq_name, label_id)
        pic2video(img_dir, video_name, fps=5)



