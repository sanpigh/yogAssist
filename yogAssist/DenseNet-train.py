from yogAssist.includes import *
from yogAssist.utils import *
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

from tensorflow.keras.applications.vgg16 import VGG16,  preprocess_input
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import DenseNet169

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
import tensorflow as tf
import tensorflow.keras as K

ImageFile.LOAD_TRUNCATED_IMAGES = True

input_shape = (IMAGE_W, IMAGE_H, NB_OF_CHANNELS)

class DenseNet_pipeline():
    def __init__(self):
       self.experiment_name = EXPERIMENT_NAME

    def set_experiment_name(self, experiment_name):
        '''defines the experiment name for MLFlow'''
        self.experiment_name = experiment_name
        
         
    def load_base_model(self):
        return DenseNet169(weights="imagenet",
                           include_top=False, 
                           input_shape=input_shape)
    def set_nontrainable_layers(self, model):
        model.trainable = False
        return model
    
    def create_model(self):
        # getting base Dense169 model and freeze its layers
        self.base_model = self.set_nontrainable_layers(self.load_base_model())
        
        # add layers
        flatten_layer_1 = layers.Flatten()
        batch_normalization_1 = layers.BatchNormalization()
        dense_layer_1 = layers.Dense(20, activation='relu')
        dropout_layer_1 = layers.Dropout(0.4)
        #batch_normalization = layers.BatchNormalization()
        dense_layer_2 = layers.Dense(10, activation='relu')
        dropout_layer_2 = layers.Dropout(0.4)
        prediction_layer = layers.Dense(20, activation='softmax')
        self.model = Sequential([
                                self.base_model,
                                flatten_layer_1,
                                batch_normalization_1,
                                dense_layer_1,
                                dropout_layer_1,
                                dense_layer_2,
                                dropout_layer_2,
                                prediction_layer
                            ])
        # compiling model
        learning_rate_ = 1e-4
        opt = optimizers.Adam(learning_rate=learning_rate_)
        self.model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=['accuracy'])
        
        self.log_model_params()
        self.log_model_optimizer("Adam", "categorical_crossentropy",str(learning_rate_), "accuracy")
    
 
        
    def run(self, epochs_= 30, batch_size_ = 124):
        # create simple pipeline model
        self.create_model()
        # log MLFlow parameters
        # create es criteria
        es = EarlyStopping(monitor='val_accuracy', mode='max', patience=5, verbose=1, restore_best_weights=True)
        # fit model
        self.history = self.model.fit(self.train_set,
                                      epochs=1,
                                      batch_size=batch_size_,
                                      callbacks=[es],
                                      validation_data=self.val_set)
        self.log_model_fit_params(str(epochs_), str(batch_size_))


    def evaluate(self):
        loss_, accuracy_ = self.model.evaluate(self.test_set)
        self.log_evaluate_metrics("accuracy", accuracy_)
        return loss_, accuracy_

    # logging to MLFlow methods
    def log_model_params(self):
        # logging layer names
        # todo: loging layer parameters
        for i, l in enumerate(self.model.layers):
            self.mlflow_log_param(f"model_add_layers_{str(i)}", l.name)
        self.mlflow_log_param("model_count_params", str(self.model.count_params()))
        
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
        
    def load_data(self):
        dataLoader = dt.DataLoader()
        self.train_set, self.val_set, self.test_set = dataLoader.train_val_test_split_ext()

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
        
    def save_model_locally(self):
        """Save the model into a .joblib format"""
        joblib.dump(self.model , 'model.joblib')
        print(colored("model.joblib saved locally", "green"))
        
    def save_history_locally(self):
        """Save the history (dict) into a .csv"""  
        # convert the history.history dict to a pandas DataFrame:     
        hist_df = pd.DataFrame(self.history) 
        #  save to csv 
        hist_csv_file = 'history.csv'
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)

if __name__ == "__main__":
    
    DenseNet_ = DenseNet_pipeline()
    # retrieving data and storing access to it in DenseNet_ instance
    DenseNet_.load_data()   
    # perform actual fitting on data - training model
    DenseNet_.run()
    DenseNet_.history
    print(f'accuracy:{DenseNet_.evaluate()}')

    #plot_model(DenseNet_.model, to_file='hrnet.png')
    webbrowser.open(get_mlflow_expirement_url(), new=2)
