
# dealing with strings, requests, regex...
import re
from urllib.parse import urlparse

import requests
import json
from tqdm.notebook import tqdm
from PIL import Image
import math

from yogAssist.includes import *
from yogAssist.params import *

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
  
# open json file and return (only first occurence...) of list of keypoints dictionnary
def read_asana_reference_file(path:str):
  with open(path) as json_file:
      data = json.load(json_file)
      return data[0]['keypoints']
    
def save_output_dictionnary(dict, path):
  with open(path, 'w') as fp:
    json.dump(dict, fp)
  
          
def decode_api_dictionnary(list_of_keypoints: list, confidence_threshold = 0.15):
  list_of_keypoints_dict = {}
  for keypoint in list_of_keypoints:
    if keypoint['score'] > confidence_threshold:
      list_of_keypoints_dict[keypoint['part']] = (keypoint['position']['x'], keypoint['position']['y'])
  return list_of_keypoints_dict

def extract_keypoints_dictionnary_from_json_api(path, confidence_threshold = 0.15):
  list_of_keypoints = read_asana_reference_file(path)
  return decode_api_dictionnary(list_of_keypoints, confidence_threshold)

# scoring maths
def compute_cosine(AB, AC):
    # cosine similarity between A and B
    cos_sim=np.dot(AB,AC)/(np.linalg.norm(AB)*np.linalg.norm(AC))
    return cos_sim

def cosin_to_degree(cosin_):
  return math.degrees(math.acos(cosin_))

def compute_cosine_sim_L1(Theta_1, Theta_2, convert_to_degree = True):
  return 180 - abs(abs(cosin_to_degree(Theta_2) - cosin_to_degree(Theta_1)) - 180)

def compute_cosine_sim_L2(Theta_1, Theta_2, convert_to_degree = True):
  return  math.pow(cosin_to_degree(Theta_2) - cosin_to_degree(Theta_1),2)

if __name__ == "__main__":
  
  #path = '/Users/gwenael-pro/code/DataGwen/le-wagon/yogAssist/assets/Virasana or Vajrasana.txt'
  path = '/Users/gwenael-pro/code/DataGwen/le-wagon/yogAssist/assets/viparita virabhadrasana or reverse warrior pose.txt'
  list_of_keypoints = read_asana_reference_file(path)
  list_of_keypoints_dict = decode_api_dictionnary(list_of_keypoints)
  for k, v in list_of_keypoints_dict.items():
    print(k)
    print(v)