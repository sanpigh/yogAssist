import yogAssist.visualizations as vis
from yogAssist.model_wrapper import ModelWrapper

import configs.draw_config as draw_config
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sys import argv

model_path = "trained_models/model11_test-15Sun1219-2101"
model_wrapper = ModelWrapper(model_path)


while True:

    image_path = input("Enter your image path: ")

    img = Image.open(image_path)

    im = np.array(img.convert('RGB'))
    skeletons = model_wrapper.process_image(im)

    skeleton_drawer = vis.SkeletonDrawer(im, draw_config)
    if len(skeletons) ==0:
        print('skeleton load failed')
    else:
        for skeleton in skeletons:
            print(skeleton.keypoints)
            skeleton.draw_skeleton(skeleton_drawer.joint_draw,
                                  skeleton_drawer.kpt_draw)
        im_tosave = Image.fromarray(im)
        path_save = "result_sekeleton_" + image_path
        im_tosave.save(path_save)
        print('image saved under ' + path_save)
        im_tosave.show()
