import tensorflow as tf
import yogAssist.post as post

import configs.post_config as post_config
import configs.keypoints_config as kpts_config
import configs.default_config as def_config
import  matplotlib.pyplot as plt

class ModelWrapper:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        
    def process_image(self, img):
        # preprocessing image
        input_img = tf.image.resize(img, (def_config.IMAGE_HEIGHT, def_config.IMAGE_WIDTH))
        input_img = tf.image.convert_image_dtype(input_img, dtype=tf.float32)
        input_img /= 255
        input_img = input_img[tf.newaxis, ...]
        # predict skeleton
        pafs, kpts = self.model.predict(input_img)
        return pafs[0], kpts[0]

