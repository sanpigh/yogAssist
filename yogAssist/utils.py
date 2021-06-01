
from yogAssist.includes import *
from yogAssist.params import *

# dealing with strings, requests, regex...
import re
from urllib.parse import urlparse

import requests
from tqdm.notebook import tqdm
from PIL import Image

 # image utils               
def get_image_format(path)->str:
  try:
    format = Image.open(path).format
    return str(format).lower()
  except:
    return None 

# system utils
def base_name(file):
  return basename(file)

def file_name(file):
  filename, fileextension = splitext(file)
  return filename 

def file_extension(file):
  filename, fileextension = splitext(file)
  return fileextension.lower() 

def listing_file_extension(extension_dict, file_list):
  for f in file_list:
    if not file_extension(f) in extension_dict.keys():
      extension_dict[file_extension(f)] = True
  return extension_dict

def couting_corrupted_file_extension(file_list):
  corrupted_count = 0
  for f in file_list:
    filename, fileextension = splitext(f)
  return corrupted_count

def list_of_classes(dataset_root_path):
    return [f for f in listdir(dataset_root_path) if isdir(join(dataset_root_path, f))]
  
def list_of_files_per_class(class_root_path):
   tmp = [join(class_root_path, f) for f in listdir(class_root_path) if isfile(join(class_root_path, f))]
   return tmp
 
def list_of_files(dataset_root_path):
  total_list_of_files = []
  classes = list_of_classes(dataset_root_path)

  for class_ in classes:
    class_dir = dataset_root_path + "/" + class_
    total_list_of_files += list_of_files(class_dir)

  return total_list_of_files

def build_dataframe_from_classes(dataset_root_path):
  total_list_of_files = []
  total_list_of_labels = []
  classes = list_of_classes(dataset_root_path)

  for class_ in classes:
    class_dir = dataset_root_path + "/" + class_
    tmp_files = [join(class_dir, f) for f in listdir(class_dir) if isfile(join(class_dir, f))]
    tmp_labels = [class_] * len(tmp_files)
    total_list_of_files += tmp_files
    total_list_of_labels += tmp_labels
  return total_list_of_labels, total_list_of_files

#MLFlow utils
def get_mlflow_expirement_url():
    mlflow.set_tracking_uri(MLFLOW_URI)
    experiment_id = MlflowClient().get_experiment_by_name(f"{EXPERIMENT_NAME}").experiment_id
    #return f"{EXPERIMENT_NAME}"
    return f"{MLFLOW_URI}/#/experiments/{experiment_id}"

