import yogAssist.visualizations as vis
from yogAssist.ModelWrapper import ModelWrapper
from yogAssist.post import *

import configs.draw_config as draw_config
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from sys import argv


model_path = "trained_models/model11_test-15Sun1219-2101"
#model_path = "trained_models/evopose2d_M_f32.h5"
model_wrapper = ModelWrapper(model_path)


while True:
    
    image_path = input("Enter your image path: ")

    img = Image.open(image_path)
    im = np.array(img.convert('RGB'))
    pafs, kps = model_wrapper.process_image(im)
    
    # init class variables
    Skeletonizer.config(KEYPOINTS_DEF, JOINTS_DEF, \
                        KEYPOINTS_HEATMAP_THRESHOLD, JOINT_ALIGNMENT_THRESHOLD)
    Skeleton.config(KEYPOINTS_DEF, JOINTS_DEF)
     # create skeletonizers and skeletons from pafs, kps
    skeletonizer_ = Skeletonizer(kps, pafs)
    skeletons = skeletonizer_.create_skeletons() 
    skeleton_drawer = vis.SkeletonDrawer(im, draw_config)
    print(os.getcwd())
    if len(skeletons) ==0:
        print('skeleton load failed')
    else:
        for skeleton in skeletons:
            print(skeleton.keypoints)
            skeleton.draw_skeleton(skeleton_drawer.joint_draw,
                                  skeleton_drawer.kpt_draw)
        im_tosave = Image.fromarray(im)
        path_save = f"{os.getcwd()}/_sekeleton_{basename(image_path)}"
        im_tosave.save(path_save)
        print('image saved under ' + path_save)
        im_tosave.show()
