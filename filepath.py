import os
from config import mode

raw_data_path = "./data"
dataset_path = "./trivago_datasets"
model_path = "./lgb_models"

train_path = os.path.join(raw_data_path, "train.csv")
test_path = os.path.join(raw_data_path, "test.csv")
metadata_path = os.path.join(raw_data_path, "item_metadata.csv")

# ground truths for validation and confirmation datasets
val_path = os.path.join(raw_data_path, "validation.csv")
conf_path = os.path.join(raw_data_path, "confirmation.csv")

if mode == "experiment":
    mode = "exp"
else:
    mode = "full"

samples_file = f"samples_{mode}.feather"
log_file = f"log_data_{mode}.feather"
exploded_file = f'log_exploded_{mode}.feather'
item_file = f"item_features_{mode}.feather"
session_file = f"session_features_{mode}.feather"
prepared_file = f"dataset_{mode}.feather"
meta_file = 'item_meta_split.feather'
gt_file = f'ground_truth_{mode}.feather'


samples_path = os.path.join(dataset_path, samples_file)
log_path = os.path.join(dataset_path, log_file)
exploded_path = os.path.join(dataset_path, exploded_file)
item_path = os.path.join(dataset_path, item_file)
session_path = os.path.join(dataset_path, session_file)
prepared_path = os.path.join(dataset_path, prepared_file)
meta_split_path = os.path.join(dataset_path, meta_file)
ground_truth_path = os.path.join(dataset_path, gt_file)
