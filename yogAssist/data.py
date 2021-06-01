import pandas as pd
from yogAssist.params import *
from yogAssist.utils import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator,\
                                                 load_img, img_to_array
from tensorflow.keras.preprocessing import image_dataset_from_directory

from PIL import Image


class DataLoader():
  
    def __init__(self, mode="local"):
        self.mode = mode
        pass
  
    def get_data_as_dataframe(self):

      dataset_root_path = f"{BASE_PATH}/{TRAIN_SUBDIR_20}"
      
      # retrieve dictionnary from classes
      labels, list_of_files_ = build_dataframe_from_classes(dataset_root_path)
      dataframe_dict = {
                        'filenames': list_of_files_,
                        'class_labels': labels
                        }
      self.df_train = pd.DataFrame(data=dataframe_dict)
      return self.df_train
  
    def create_path_from_gcp(self):
        """method to get the training data  from google cloud bucket"""
        # Add Client() here
        client = storage.Client()
        bucket = client.get_bucket({BUCKET_NAME})
        self.dataset_train_root_path = f"gs://{BUCKET_NAME}/{BUCKET_PROJECT}/{BUCKET_TRAIN_DATA_PATH}"
        self.dataset_test_root_path = f"gs://{BUCKET_NAME}/{BUCKET_PROJECT}/{BUCKET_TRAIN_DATA_PATH}"
        
    def create_path_from_local(self):
        #self.dataset_train_root_path = f"{BASE_PATH}/{TRAIN_SUBDIR}"
        #self.dataset_test_root_path = f"{BASE_PATH}/{TEST_SUBDIR}"
        self.dataset_train_root_path = f"{BASE_PATH}/{TRAIN_SUBDIR_20}"
        self.dataset_test_root_path = f"{BASE_PATH}/{TEST_SUBDIR_20}"
        
        
        
    def init_data_path(self):
        if self.mode == "local":
            self.create_path_from_local()
            
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
        
        self.train_set = self.train_gen.flow_from_directory(self.dataset_train_root_path,
                                            target_size=(image_width,image_height),
                                            batch_size=batch_size,
                                            class_mode='categorical',
                                            subset='training')
        self.test_set = self.test_gen.flow_from_directory(self.dataset_test_root_path,
                                            target_size=(image_width,image_height),
                                            batch_size=batch_size,
                                            class_mode='categorical')
        self.val_set = self.train_gen.flow_from_directory(self.dataset_train_root_path,
                                            target_size=(image_width,image_height),
                                            batch_size=batch_size,
                                            class_mode='categorical',
                                            subset='validation')
        
    def image_dataset_from_directory(self,
                                    batch_size=32, 
                                    image_width= 384, 
                                    image_height= 384):
        
        print("Create train, test and validation set...")

        self.train_set = image_dataset_from_directory(self.dataset_train_root_path,
                                                     image_size=(image_width,image_height),
                                                     batch_size=batch_size,
                                                     labels="inferred",
                                                     label_mode='categorical',
                                                     class_names = list_of_classes(self.dataset_train_root_path),
                                                     validation_split=0.1,
                                                     seed=1337,
                                                     subset="training")

        self.val_set = image_dataset_from_directory(self.dataset_train_root_path,
                                                    image_size=(image_width,image_height),
                                                    batch_size=batch_size,
                                                    labels="inferred",
                                                    label_mode='categorical',
                                                    class_names = list_of_classes(self.dataset_train_root_path),
                                                    validation_split=0.1,
                                                    seed=1337,
                                                    subset="validation")    
        
        self.test_set = image_dataset_from_directory(self.dataset_test_root_path,
                                                    image_size=(image_width,image_height),
                                                    batch_size=batch_size,
                                                    labels="inferred",
                                                    label_mode='categorical',
                                                    shuffle=False,
                                                    class_names = list_of_classes(self.dataset_test_root_path))
        
    def image_datasets_from_folder_tree(self,
                                        img_to_nparray_flag=False,
                                        batch_size=32, 
                                        image_width= 384, 
                                        image_height= 384):
        tree_ds = {}
        tree_files_per_class = {}
        self.init_data_path()
        classes_ = list_of_classes(self.dataset_train_root_path)
        for c_ in classes_:
            files = list_of_files_per_class(f"{self.dataset_train_root_path}/{c_}")    
            training_data = []
            for img in files:
                
                if img_to_nparray_flag == True:
                    pic = Image.open(img).convert('RGB')
                    pic = pic.resize((image_width,image_height), Image.BILINEAR)
                    pix_array = np.array(pic).reshape(1, image_width, image_width, 3)
                    training_data.append(pix_array)
                else:
                    img = load_img(img, target_size=(image_width,image_height))
                    img_array = img_to_array(img)
                    #img_batch = np.expand_dims(img_array, axis=0)
                    training_data.append(img_array)
                    
            tree_ds[c_ ] = training_data
            tree_files_per_class[c_ ] = files
        return tree_ds, tree_files_per_class
            
    def train_val_test_split(self, batch_size=BATCH_SIZE,  image_width= IMAGE_W, image_height= IMAGE_H):
        self.init_data_path()
        self.make_image_data_generator()
        self.flow_from_directory(batch_size, image_width, image_height)
        return self.train_set, self.val_set, self.test_set
    
    def train_val_test_split_ext(self, batch_size=BATCH_SIZE,  image_width= IMAGE_W, image_height= IMAGE_H):
        self.init_data_path()
        self.image_dataset_from_directory(batch_size, image_width, image_height)
        return self.train_set, self.val_set, self.test_set
    
if __name__ == "__main__":
    
    dl = DataLoader()
    #dl.create_path_from_gcp()
    df = dl.get_data_as_dataframe()
    dl.image_datasets_from_folder_tree()
    print(df.head())
