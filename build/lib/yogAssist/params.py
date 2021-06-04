BASE_PATH = 'raw_data'
MODEL_PATH = 'trained_models'
TRAIN_SUBDIR='TRAIN'
TEST_SUBDIR='TEST'
TRAIN_SUBDIR_20='TRAIN-20-classes'
TEST_SUBDIR_20='TEST-20-classes'

# Dense169 hyper-parameters
BATCH_SIZE = 32

IMAGE_W= 384
IMAGE_H= 384
NB_OF_CHANNELS=3

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'Classification-yoga-poses'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v0.1'

### MLFLOW configuration - - - - - - - - - - - - - - - - - - -
MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = f"[FR] [PARIS] [YogAssist] {MODEL_NAME}-{MODEL_VERSION}"

### DATA & MODEL LOCATIONS  - - - - - - - - - - - - - - - - - - -
PATH_TO_LOCAL_MODEL = f"{MODEL_NAME}-{MODEL_VERSION}.joblib"


### HANDLING GCP STORAGE AND TRAINING
BUCKET_NAME = 'data-589-datagwen'
BUCKET_PROJECT= 'yogAssist'
BUCKET_TRAIN_DATA_PATH = 'data/dataset/TRAIN'
BUCKET_TRAIN_DATA_PATH = 'data/dataset/TRAIN'








