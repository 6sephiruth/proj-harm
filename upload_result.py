import os
import random
import time
import pickle
import shap
import pandas as pd

import pyarrow.parquet as pq
from pyarrow import csv
from sklearn.neural_network import MLPClassifier

from utils import *

def main():

    # designate gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    # enable memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    for d in physical_devices:
        tf.config.experimental.set_memory_growth(d, True)

    SEED = 0

    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # Models
    raw_models = pickle.load(open('./models/raw_model.pt','rb'))
    raw_columns = pickle.load(open('./models/raw_columns','rb'))

    shap_models = pickle.load(open('./models/shap_models.pt','rb'))
    shap_columns = pickle.load(open('./models/shap_columns','rb'))

    # Datasets
    analysis_datasets = pd.read_parquet('./datasets/collectResult/21000-22300.parquet')
    
    raw_analysis_datasets = pd.DataFrame(analysis_datasets, columns=raw_columns)
    raw_analysis_datasets.fillna(0, inplace=True)

    shap_analysis_datasets = pd.DataFrame(analysis_datasets, columns=shap_columns)
    shap_analysis_datasets.fillna(0, inplace=True)

    file_id = np.array(analysis_datasets['file_id'])
    raw_model_label = raw_models.predict(raw_analysis_datasets)
    shap_model_label = shap_models.predict(shap_analysis_datasets)
    label_prob = np.max(shap_models.predict_proba(shap_analysis_datasets), axis=1)

    #### SHAP
    # explainer
    explainer = shap.TreeExplainer(raw_models, seed=SEED)
    shap_values = explainer.shap_values(raw_analysis_datasets)

    for i in range(len(raw_analysis_datasets)):    
        
        part_key = np.array(raw_columns)[np.argsort(shap_values[raw_model_label[i]])[i][-5:]]
        part_value = np.sort(shap_values[raw_model_label[i]])[i][-5:]
                
        part_position = np.argsort(shap_values[raw_model_label[i]])[i][-5:]

        a1, a2, a3, a4, a5 = round(shap_values[0][i][part_position[0]], 6), round(shap_values[0][i][part_position[1]], 6), round(shap_values[0][i][part_position[2]], 6), round(shap_values[0][i][part_position[3]], 6), round(shap_values[0][i][part_position[4]], 6)
        b1, b2, b3, b4, b5 = round(shap_values[1][i][part_position[0]], 6), round(shap_values[1][i][part_position[1]], 6), round(shap_values[1][i][part_position[2]], 6), round(shap_values[1][i][part_position[3]], 6), round(shap_values[1][i][part_position[4]], 6)
        c1, c2, c3, c4, c5 = round(shap_values[2][i][part_position[0]], 6), round(shap_values[2][i][part_position[1]], 6), round(shap_values[2][i][part_position[2]], 6), round(shap_values[2][i][part_position[3]], 6), round(shap_values[2][i][part_position[4]], 6)

        white_value = f"{a1},{a2},{a3},{a4},{a5}"
        gamble_value = f"{b1},{b2},{b3},{b4},{b5}"
        ad_value = f"{c1},{c2},{c3},{c4},{c5}"
        
        total_value = f"{white_value},{gamble_value},{ad_value}"

        d1, d2, d3, d4, d5 = round(part_value[0],3), round(part_value[1],3), round(part_value[2],3), round(part_value[3],3), round(part_value[4],3)

        ff_text = f"{part_key[0]},{part_key[1]},{part_key[2]},{part_key[3]},{part_key[4]}"
        ff_value = f"{d1},{d2},{d3},{d4},{d5}"

        if i == 0:
            xai_top_text = np.array([ff_text])
            xai_top_value = np.array([ff_value])
            xai_total_value = np.array([total_value])
        else:
            xai_top_text = np.concatenate((xai_top_text, np.array([ff_text])), axis=0)
            xai_top_value = np.concatenate((xai_top_value, np.array([ff_value])), axis=0)
            xai_total_value = np.concatenate((xai_total_value, np.array([total_value])), axis=0)
        print(i)

    result = pd.DataFrame({'id':file_id,
                            'raw_model_label':raw_model_label,
                            'shap_model_label':shap_model_label,
                            'label_prob':label_prob,
                            'xai_top_text':xai_top_text,
                            'xai_total_value':xai_total_value,
                            'xai_top_value':xai_top_value,
                            'xai_total_value':xai_total_value
                            })

    result.to_csv("./21000-22300.csv", mode='w')

# run main
if __name__ == "__main__":
    main()
