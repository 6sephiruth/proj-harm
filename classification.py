import os
import random
import time
import pickle
import shap

import pyarrow.parquet as pq
from pyarrow import csv
from sklearn.neural_network import MLPClassifier

from utils import *

def main():


    # designate gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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

    white_dataset = load_total_dataframe('./datasets/white/')
    white_dataset['CSRC_label'] = 0
    gamble_dataset = load_total_dataframe('./datasets/gamble/')
    gamble_dataset['CSRC_label'] = 1
    ad_dataset = load_total_dataframe('./datasets/ad/')
    ad_dataset['CSRC_label'] = 2

    labeled_dataset = load_total_dataframe('./datasets/total/')

    correct_dataset_new = pd.read_parquet('./datasets/collectResult0503.parquet')

    # print("Correct New 데이터셋에서")
    # print(f"정상 개수: {len(np.where(np.array(correct_dataset_new['CSRC_label']) == 0)[0])}, \
    #         도박 개수: {len(np.where(np.array(correct_dataset_new['CSRC_label']) == 1)[0])}, \
    #         광고 개수: {len(np.where(np.array(correct_dataset_new['CSRC_label']) == 2)[0])}")
    # print(correct_dataset_new.shape)
    # exit()

    init_train = pd.concat([white_dataset, gamble_dataset, ad_dataset, labeled_dataset])
    init_train.fillna(0, inplace=True)

    x_init_train = init_train.drop('CSRC_label', axis=1).drop('file_id', axis=1)
    y_init_train = init_train['CSRC_label']

    raw_columns = x_init_train.columns

    ##### training #####
    try:
        # load model if possible
        raw_models = pickle.load(open('./models/base_model.pt','rb'))
    except:
        raw_models = xgb.XGBClassifier(n_estimators=200,
                                max_depth=10,
                                learning_rate=0.5,
                                min_child_weight=0,
                                tree_method='gpu_hist',
                                sampling_method='gradient_based',
                                reg_alpha=0.2,
                                reg_lambda=1.5,
                                random_state=SEED)

        raw_models.fit(x_init_train, y_init_train)
        pickle.dump(raw_models, open('./models/base_model.pt','wb'))

    # comparison with model feature importance
    feat_importance = list(raw_columns[raw_models.feature_importances_ > 0])
    x_import = x_init_train[feat_importance]
    y_import = y_init_train

    #### SHAP
    # explainer
    explainer = shap.TreeExplainer(raw_models, seed=SEED)
    shap_values = explainer.shap_values(x_init_train)

    # filtering mode
    #FILTER = "by-order"         # 각 클래스별 top 100 키워드 추출
    FILTER = "by-thresh"        # 각 클래스별 SHAP이 0보다 큰 키워드 추출

    # placeholder for feature sets
    feat_shap = []

    n_class = 3
    for cls in range(n_class):
        attr = shap_values[cls]

        # calculate mean(|SHAP values|) for each class
        avg_shap = np.abs(attr).mean(0)
        l = len(avg_shap)

        # filtering by ordering
        if FILTER == 'by-order':
            idxs = np.argpartition(avg_shap, l-100)[-100:]
            keywords = set(raw_columns[idxs])

        # filtering by thresholding
        elif FILTER == 'by-thresh':
            keywords = set(raw_columns[avg_shap > 0])

        feat_shap.append(keywords)

    # keywords from shap
    from functools import reduce
    feat_shap_all = list(reduce(set.union, feat_shap))

    # filter columns
    x_shap_train = x_init_train[feat_shap_all]
    shap_columns = x_shap_train.columns

    try:
        # load model if possible
        shap_models = pickle.load(open('shap_models.pt','rb'))

    except:
        shap_models = xgb.XGBClassifier(n_estimators=200,
                                max_depth=10,
                                learning_rate=0.5,
                                min_child_weight=0,
                                tree_method='gpu_hist',
                                sampling_method='gradient_based',
                                reg_alpha=0.2,
                                reg_lambda=1.5,
                                random_state=SEED)

        shap_models.fit(x_shap_train, y_init_train)
        pickle.dump(shap_models, open('./models/shap_models.pt','wb'))

    # y_pred = shap_models.predict(x_shap_train)
    # accuracy = accuracy_score(y_init_train, y_pred)
    # print("Train accuracy: %.2f" % (accuracy * 100.0))

    # x_test = pd.DataFrame(correct_dataset_new, columns=shap_columns)
    # x_test.fillna(0, inplace=True)
    # y_test = correct_dataset_new['CSRC_label']

    # y_pred = shap_models.predict(x_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # print("Train accuracy: %.2f" % (accuracy * 100.0))

# run main
if __name__ == "__main__":
    main()