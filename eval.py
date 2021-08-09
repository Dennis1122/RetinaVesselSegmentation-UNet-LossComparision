import os
import tensorflow as tf
import config
from utils.metrics import iou, dice
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import classification_report, roc_auc_score, matthews_corrcoef, jaccard_score, accuracy_score
from utils.augmentations import Val_Generator
from utils import dataset_utils as ds
import pandas as pd
from datetime import datetime as dt

# # Configurations
EXP_NAME = config.EXP_NAME
loss_function = config.loss_function
model_path = f"{EXP_NAME}/training/model.h5"
dataset_path = os.path.join("new_data", "test")
result_folder = f"{EXP_NAME}/results"
score_path = f"{EXP_NAME}/score.csv"

# Global variables
train_path = config.train_path
test_path = config.test_path
batch_img_dim = config.batch_img_dim
batch_msk_dim = config.batch_msk_dim

# Load dataset
train_images_files, train_mask_files, test_images_files, test_mask_files = ds.load_dataset(train_path, test_path)

# load model
with CustomObjectScope({'iou': iou, 'dice': dice, 'loss': loss_function}):
    model = tf.keras.models.load_model(model_path)

test_generator = Val_Generator(test_images_files, test_mask_files, batch_size=20,
                               img_dim=(batch_img_dim[1], batch_img_dim[2]), augmentation=True)

for x_test, y_test in test_generator:
    break

y_pred = model.predict(x_test)

y_true = (y_test > 0.5).flatten()
y_pred = (y_pred > 0.5).flatten()

report = classification_report(y_true, y_pred, output_dict=True)

Precision = report['True']['precision']
Recall = report['True']['recall']
F1_score = report['True']['f1-score']

Sensitivity = Recall
Specificity = report['False']['recall']

IOU = (Precision * Recall) / (Precision + Recall - Precision * Recall)
AUC = roc_auc_score(y_true.flatten(), y_pred.flatten())
MCC = matthews_corrcoef(y_true.flatten(), y_pred.flatten())

jac_value = jaccard_score(y_true.flatten(), y_pred.flatten())
acc_value = accuracy_score(y_true.flatten(), y_pred.flatten())

print(classification_report(y_true, y_pred))

# create or load CSV report
try:
    result_df = pd.read_csv('results.csv')
except FileNotFoundError:
    print('File Not Found - creating new file')
    result_df = pd.DataFrame(
        columns=["Date", "Experiment Name", "F1", "AUC", "MCC", "IOU", "Acc", "Jaccard", "Recall", "Precision",
                 "Sensitivity", "Specificity"])

# update results to dataframe
result_list = [str(dt.now()), EXP_NAME, F1_score, AUC, MCC, IOU, acc_value, jac_value, Recall, Precision, Sensitivity,
               Specificity]
result_df.loc[len(result_df)] = result_list

# save dataframe as csv
result_df.to_csv('results.csv', index=False)