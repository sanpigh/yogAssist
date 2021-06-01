from yogAssist.utils import *
from yogAssist.includes import *

from yogAssist.params import *
import yogAssist.data as dt

from tensorflow.keras.utils import plot_model

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.applications import ( VGG19, DenseNet169 )

from tensorflow.keras.applications.vgg16 import VGG16,  preprocess_input

from tensorflow.python.keras.layers import (
    Input,
    GlobalAveragePooling2D,
    Dense,
    MaxPooling2D,
    UpSampling2D,
    AveragePooling2D,
    Conv2D,
    BatchNormalization,
    Concatenate,
    Activation,
    Flatten,
    Add,
    Multiply,
    Reshape,
    Lambda,
    Dropout,
    Layer
)

from tensorflow.python.keras.regularizers import l2

ImageFile.LOAD_TRUNCATED_IMAGES = True

import tensorflow as tf
import tensorflow.keras as K

input_shape = (IMAGE_W, IMAGE_H, NB_OF_CHANNELS)

class HRNet_pipeline(K.layers.Layer):
    
    def __init__(self, blocks=3, reduction_ratio=4, init_filters=64, expansion=4, training=True):
        self.blocks = blocks
        self.training = training
        self.reduction_ratio = reduction_ratio
        self.init_filters = init_filters
        self.expansion = expansion
        print("Expansion part has not been implemented.")
        self.experiment_name = EXPERIMENT_NAME

    def set_experiment_name(self, experiment_name):
        '''defines the experiment name for MLFlow'''
        self.experiment_name = experiment_name

        
    def load_data(self):
        dataLoader = dt.DataLoader()
        self.train_set, self.val_set, self.test_set = dataLoader.train_val_test_split()
        
    def first_layer(self, x, scope):
            with tf.name_scope(scope):
                x = Conv2D(filters=self.init_filters, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_regularizer=l2(1e-4))(x)
                x = BatchNormalization(axis=-1)(x, training=self.training)
                x = Conv2D(filters=self.init_filters, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_regularizer=l2(1e-4))(x)
                norm = BatchNormalization(axis=-1)(x, training=self.training)
                act = Activation('relu')(norm)
                return act
            
    def bottleneck_block(self, x, filters, strides, scope):
        with tf.name_scope(scope):
            x = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(1e-4))(x)
            x = BatchNormalization(axis=-1)(x, training=self.training)
            x = Activation('relu')(x)
            x = Conv2D(filters=filters, kernel_size=(3, 3), strides=strides, padding='same', kernel_regularizer=l2(1e-4))(x)
            x = BatchNormalization(axis=-1)(x, training=self.training)
            x = Activation('relu')(x)
            x = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(1e-4))(x)
            x = BatchNormalization(axis=-1)(x, training=self.training)
            return x
        
    def squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
            with tf.name_scope(layer_name):
                squeeze = GlobalAveragePooling2D()(input_x)
                excitation = Dense(units=out_dim / ratio)(squeeze)
                excitation = Activation('relu')(excitation)
                excitation = Dense(units=out_dim)(excitation)
                excitation = Activation('sigmoid')(excitation)
                excitation = Reshape([1, 1, out_dim])(excitation)
                scale = Multiply()([input_x, excitation])
                return scale
            
    def residual_layer(self, out_dim, scope, first_layer_stride=(2, 2), res_block=None):
        if res_block is None:
            res_block = self.blocks
        def f(input_x):
            # split + transform(bottleneck) + transition + merge
            # input_dim = input_x.get_shape().as_list()[-1]
            for i in range(res_block):
                if i == 0:
                    strides = first_layer_stride
                    # filters = input_x.get_shape().as_list()[-1]
                else:
                    strides = (1, 1)
                    # filters = out_dim
                x = self.bottleneck_block(input_x, filters=out_dim, strides=strides, scope='bottleneck_' + str(i))
                x = self.squeeze_excitation_layer(x, out_dim=x.get_shape().as_list()[-1], ratio=self.reduction_ratio, layer_name='squeeze_layer_' + str(i))
                if i != 0:  # Leave the first block without residual connection due to unequal shape
                    x = Add()([input_x, x])
                x = Activation('relu')(x)
                input_x = x
            return input_x
        return f
    
    def multi_resolution_concat(self, maps, filter_list, scope='multi_resolution_concat'):
        fuse_layers = []
        print('Input', maps)
        with tf.name_scope(scope):
            for idx, _ in enumerate(maps):
                fuse_list = []
                for j in range(len(maps)):
                    x = maps[j]
                    # Upsamples, high resolution first
                    if j < idx:
                        # Downsamples, high resolution first
                        for k in range(idx - j):
                            x = Conv2D(filter_list[idx], kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
                            x = BatchNormalization(axis=-1)(x, training=self.training)
                            if k == idx - j - 1:
                                x = Activation('relu')(x)
                    elif j == idx:
                        # Original feature map
                        pass
                    elif j > idx:
                        for k in range(j - idx):
                            x = Conv2D(filter_list[idx], kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
                            x = BatchNormalization(axis=-1)(x, training=self.training)
                            x = UpSampling2D(size=(2, 2))(x)
                    else:
                        raise ValueError()
                    fuse_list.append(x)
                    print(idx, j, maps[j])
                    print(filter_list[idx], x)
                if len(fuse_list) > 1:
                    concat = Add()(fuse_list)
                    x = Activation('relu')(concat)
                    fuse_layers.append(x)
                    # print(‘Assemble’, concat)
                else:
                    fuse_layers.append(fuse_list[0])
                    # print(‘Assemble O’, fuse_list[0])
            print('Out', fuse_layers)
        return fuse_layers

    def extract_multi_resolution_feature(self, repetitions=3):
        def f(input_x):
            x = self.first_layer(input_x, scope='first_layer')
            features = []
            filters = self.init_filters
            self.filter_list = [filters]
            # First Layer consumed one stage
            for i in range(repetitions):
                print("\nBuilding ... %d/%d" % (i, repetitions))
                # Get Downsample
                scope = 'stage_%d' % (i + 1)
                if i == 0:
                    down_x = self.residual_layer(filters, scope=scope, first_layer_stride=(2, 2), res_block=self.blocks)(x)
                else:
                    down_x = self.residual_layer(filters, scope=scope, first_layer_stride=(2, 2), res_block=self.blocks)(features[-1])
                features.append(down_x)
                # Get concatenated feature maps
                out_maps = self.multi_resolution_concat(features, self.filter_list)
                features = []
                print('Identity Mapping:')
                # Residual connection with 3x3 kernel, 1x1 stride with same number of filters
                for idx, (fm, num_filter) in enumerate(zip(out_maps, self.filter_list)):
                    x = Lambda(lambda x: x, output_shape=x.get_shape().as_list())(fm)
                    print(idx, x)
                    features.append(x)
                filters *= 2
                self.filter_list.append(filters)
            return features
        return f
    
    def make_classification_head(self, feature_maps, filter_list):
        previous_fm = None
        for idx, fm in enumerate(feature_maps):
            if previous_fm is None:
                previous_fm = fm
                continue
            # if idx == len(feature_maps):
            #     # The final feature map no need to add
            #     continue
            print(previous_fm.get_shape().as_list(), fm.get_shape().as_list(), filter_list[idx], filter_list)
            x = Conv2D(filter_list[idx], kernel_size=(3, 3), strides=(2, 2), padding='same')(previous_fm)
            x = BatchNormalization(axis=-1)(x, training=self.training)
            x = Activation('relu')(x)
            previous_fm = Add()([fm, x])
        return previous_fm
    
    def build(self, input_shape, num_output, repetitions=3):
        input_x = Input(shape=input_shape)
        feature_maps = self.extract_multi_resolution_feature(repetitions=repetitions)(input_x)
        x = self.make_classification_head(feature_maps, self.filter_list)
        x = Conv2D(filters=x.get_shape().as_list()[-1] * 2, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization(axis=-1)(x, training=self.training)
        x = Activation('relu')(x)
        x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)
        x = Dense(units=num_output,
                  name='final_fully_connected',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4),
                  activation='softmax')(x)
        self.model = Model(inputs=input_x, outputs=x)
        self.log_model_params()
        self.log_model_params_from_attr("repetitions", repetitions)
        return self.model 
    
    def compile(self):
        learning_rate_ = 1e-4
        opt = optimizers.Adam(learning_rate=learning_rate_)
        self.model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=['accuracy'])
        self.log_model_optimizer("Adam", "categorical_crossentropy",str(learning_rate_), "accuracy")
        
    
    def run(self, epochs_= 1, batch_size_ = 248):

        # log MLFlow parameters
        #self.mlflow_log_param("model", self.model.summary())
        # create es criteria
        es = EarlyStopping(monitor='val_accuracy', mode='max', patience=5, verbose=1, restore_best_weights=True)
        # fit model
        self.history = self.model.fit(self.train_set,
                                      epochs= epochs_,
                                      batch_size= batch_size_,
                                      callbacks=[es],
                                      validation_data=self.val_set)
        self.log_model_fit_params(str(epochs_), str(batch_size_))
        
    def evaluate(self):
        loss_, accuracy_ = self.model.evaluate(self.test_set)
        self.log_evaluate_metrics("accuracy", accuracy_)
        return loss_, accuracy_
    
    
    # MLFlow methods
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def log_model_params(self):
        # logging layer names
        # todo: loging layer parameters
        for i, l in enumerate(self.model.layers):
            self.mlflow_log_param(f"model_add_layers_{str(i)}", l.name)
        self.mlflow_log_param("model_count_params", str(self.model.count_params()))
        
    def log_model_params_from_attr(self, name, val): 
        self.mlflow_log_param(name, val)
        
    def log_model_optimizer(self, opt, loss, lr, metrics):
        # log optimizer parameters
        self.mlflow_log_param("model_optimizer",opt)
        self.mlflow_log_param("model_loss",loss)
        self.mlflow_log_param("model_optimizer_learning_rate", lr, )
        self.mlflow_log_param("model_metrics_metrics", metrics)
        
    def log_model_fit_params(self, epochs_, batch_size_):
        # log model fit parameters
        self.mlflow_log_param("model_fit_epochs", epochs_)
        self.mlflow_log_param("model_fit_batch_size", batch_size_)
    
    def log_evaluate_metrics(self, type, value):
        self.mlflow_log_metric(type, value)


def get_model_memory_usage(batch_size, model):
        import numpy as np
        import tensorflow.keras.backend as K
        shapes_mem_count = 0
        for l in model.layers:
            single_layer_mem = 1
            for s in l.output_shape:
                if s is None:
                    continue
                single_layer_mem *= s
            shapes_mem_count += single_layer_mem
        trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
        non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])
        number_size = 4.0
        if K.floatx() == 'float16':
            number_size = 2.0
        if K.floatx() == 'float64':
            number_size = 8.0
        total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
        gbytes = np.round(total_memory / (1024.0 ** 3), 3)
        return gbytes
    
if __name__ == "__main__":
        HRNet_ = HRNet_pipeline(blocks=3,
                                reduction_ratio=4,
                                init_filters=32,
                                training=True)
        HRNet_.load_data()
        HRNet_.build(input_shape= input_shape, num_output=20, repetitions=2)
        HRNet_.compile()
        HRNet_.run()
        #print(get_model_memory_usage(1, model))
        HRNet_.history
        print(f'accuracy:{HRNet_.evaluate()}')
        plot_model(HRNet_.model, to_file='hrnet.png')
        webbrowser.open(get_mlflow_expirement_url(), new=2)


