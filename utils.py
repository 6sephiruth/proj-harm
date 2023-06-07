import os
import schedule
import time
import pickle

from functools import reduce
from collections import Counter
from pyarrow import csv

import pandas as pd
import tensorflow as tf
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import shap

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

np.random.seed(0)

def load_total_dataframe(dir_path):
    '''
    폴더 내 모든 file list를 불러 온 뒤,
    dataFrame을 하나로 묶는 과정
    '''

    file_list = []

    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            file_list.append(dir_path + path)

    total_df = pd.DataFrame()

    for each_parquet in file_list:
        
        dataset = pd.read_parquet(each_parquet)
        dataset = cleaning_dataset(dataset)
        total_df = pd.concat([total_df, dataset])
        total_df = total_df.fillna(0)
    
    # 중복 제거(colums(속성)가 전부 같은것들)
    total_df = total_df.drop_duplicates(total_df.columns)

    # 키워드가 1개인 것들은 전부 삭제
    cols_len = np.array([len(i) for i in total_df.columns])
    total_df = total_df.drop(total_df.columns[np.where(cols_len == 1)[0]], axis=1)

    return total_df ###################################### 추후 수정

def cleaning_dataset(dataset):
    """
    데이터셋 내 에서 불필요한 속성 지우고,
    가로, 세로 sum 0인 것들 삭제
    """
    try:
        dataset = pd.DataFrame(dataset.drop('Unnamed: 0', axis=1))
        dataset = pd.DataFrame(dataset.drop('Unnamed: 0.1', axis=1))
        dataset = pd.DataFrame(dataset.drop('Unnamed: 0.2', axis=1))
        dataset = pd.DataFrame(dataset.drop('Unnamed: 0.3', axis=1))
        dataset = pd.DataFrame(dataset.drop('', axis=1))
        dataset = pd.DataFrame(dataset.drop(' ', axis=1))

    except:
        pass
    dataset = dataset.loc[(dataset.sum(axis=1) != 0), (dataset.sum(axis=0) != 0)]
    return dataset

# def cols_aug(x_dataset, cols_list, noise_type, STD):
#     np.random.seed(0)

#     # STD = 0.2

#     noise_df = pd.DataFrame()

#     for i in range(len(cols_list)):

#         if noise_type == "gaussian":
#             ist_0 = x_dataset[cols_list[i]].mean()
#             ist_1 = x_dataset[cols_list[i]].std() * STD
#         elif noise_type == "minmax":
#             ist_0 = x_dataset[cols_list[i]].min()
#             ist_1 = x_dataset[cols_list[i]].max()
#         elif noise_type == "median":
#             ist_0 = np.meadian(x_dataset[cols_list[i]])
#             ist_1 = x_dataset[cols_list[i]].std() * STD
        
#         noise_data = pd.DataFrame(np.random.normal(ist_0, ist_1, size=len(x_dataset)), columns=[cols_list[i]])
#         noise_df = pd.concat([noise_df, noise_data], axis=1)

#     # aug_dataset = x_dataset.reset_index().add(noise_df, fill_value=0)

#     print(noise_df.shape)
#     print(x_dataset.shape)
#     print(aug_dataset.shape)
#     exit()
    
#     # return aug_dataset




# def choice_aug_dataset(model, x_aug_dataset, y_aug_dataset):

#     confidence_score = model.predict_proba(x_aug_dataset)

#     score_label_confidence = []

#     for i in range(len(x_aug_dataset)):

#         score_label_confidence.append(confidence_score[i][np.array(y_aug_dataset)[i]])

#     x_select_dataset = pd.DataFrame(np.array(x_aug_dataset)[np.where(np.array(temp) > 0.95)], columns=x_aug_dataset.columns)
#     y_select_dataset = pd.DataFrame(np.array(y_aug_dataset)[np.where(np.array(temp) > 0.95)])

#     return x_select_dataset, y_select_dataset


# def model_acc(model, x_data, y_data):

#     pred = model.predict(x_data)
#     pred = accuracy_score(pred, y_data)

#     print(pred)

# def model_evaluate(model, x_data, y_data):

#     y_pred = model.predict(x_data)
#     accuracy = accuracy_score(y_data, y_pred) * 100.0
#     print("Accuracy: %.2f" % (accuracy))
#     print("-----------------------------")

#     return accuracy


# def aug_dataset_evaluate(model, org_dataset, aug_dataset):
#     print()