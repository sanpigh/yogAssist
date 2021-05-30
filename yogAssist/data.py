import pandas as pd
from yogAssist.params import *
from yogAssist.utils import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator,\
                                                 load_img, img_to_array
            
class DataLoader():
  
    def __init__(self):
        pass
  
    def get_data_as_dataframe(self):

      dataset_root_path = f"{BASE_PATH}/{TRAIN_SUBDIR}"
      
      # retrieve dictionnary from classes
      labels, list_of_files_ = build_dataframe_from_classes(dataset_root_path)
      dataframe_dict = {
                        'filenames': list_of_files_,
                        'class_labels': labels
                        }
      self.df_train = pd.DataFrame(data=dataframe_dict)
      return self.df_train
  
    def make_image_data_generator(self):
        print("Data preprocessing...")
        self.train_gen = ImageDataGenerator(rescale=1./255,
                                            shear_range=0.2,
                                            zoom_range=0.2,
                                            width_shift_range=0.12,
                                            height_shift_range=0.12,
                                            horizontal_flip=True,
                                            rotation_range=45,
                                            validation_split=0.1)
        self.test_gen = ImageDataGenerator(rescale=1./255)

    def flow_from_directory(self, batch_size=32, 
                                  image_width= 384, 
                                  image_height= 384):
        
        print("Create train, test and validation set...")
        
        dataset_train_root_path = f"{BASE_PATH}/{TRAIN_SUBDIR}"
        dataset_test_root_path = f"{BASE_PATH}/{TEST_SUBDIR}"
        
        self.train_set = self.train_gen.flow_from_directory(dataset_train_root_path,
                                            target_size=(image_width,image_height),
                                            batch_size=batch_size,
                                            class_mode='categorical',
                                            subset='training')
        self.test_set = self.test_gen.flow_from_directory(dataset_test_root_path,
                                            target_size=(image_width,image_height),
                                            batch_size=batch_size,
                                            class_mode='categorical')
        self.val_set = self.train_gen.flow_from_directory(dataset_train_root_path,
                                            target_size=(image_width,image_height),
                                            batch_size=batch_size,
                                            class_mode='categorical',
                                            subset='validation')
    def train_val_test_plit(self):
        self.make_image_data_generator()
        self.flow_from_directory(BATCH_SIZE, IMAGE_W, IMAGE_H)
        return self.train_set, self.val_set, self.test_set
    
if __name__ == "__main__":
    
    dl = DataLoader()
    df = dl.get_data_as_dataframe()
    dl.make_image_data_generator()
    dl.flow_from_directory()
    print(df.head())
