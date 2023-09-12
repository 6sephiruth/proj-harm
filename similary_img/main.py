import os
import pymysql
import cv2
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.preprocessing import minmax_scale
from sklearn.cluster import KMeans
import seaborn as sns

host='143.248.38.246'
user='root'
password='Next_lab2!'
db='hamfulsite'
port=13306

# for k in range(1,5):

#     big_arr = pickle.load(open(f'./big_arr_3000','rb'))
#     big_arr = minmax_scale(big_arr)
#     df = pd.DataFrame(big_arr, columns=['x', 'y'])
#     estimator = KMeans(n_clusters = k)
#     cluster_ids = estimator.fit_predict(df[['x', 'y']])

#     plt.scatter(df['x'], df['y'], c=cluster_ids)
#     plt.show()
#     plt.savefig(f'./out_{k}.png')

# exit()

connection = pymysql.connect(host=host, user=user, password=password, db=db, port=port, charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
cur = connection.cursor()    
sql_check = "SELECT id FROM collectResult where flag = 6 and raw_model_label is not null"
cur.execute(sql_check)
checkResult = cur.fetchall()
connection.close()

for idx, row in enumerate(checkResult):
    pid = row['id']
    # img = Image.open(f'../../../nas_server/collectResult/{pid}/img/main.png')
    img = cv2.imread(f'../../../nas_server/collectResult/{pid}/img/main.png')

    # img = img.show() # 이미지 확인
    print(img.shape)
    # img.save(f'./img/{idx}.png')
    # src = cv2.imread(f'../../../nas_server/collectResult/{pid}/img/main.png', cv2.IMREAD_GRAYSCALE)
    # print(src.shape)

    if idx == 10:
        exit()


exit()

# try:
#     adversarial_list = pickle.load(open(f'./big_arr','rb'))
# except:
for idx, row in enumerate(checkResult):
    pid = row['id']
    # img = Image.open(f'../../../nas_server/collectResult/{pid}/img/main.png')
    # img.show() # 이미지 확인
    # img.save('./1.png')
    src = cv2.imread(f'../../../nas_server/collectResult/{pid}/img/main.png', cv2.IMREAD_GRAYSCALE)
    print(src.shape)
    exit()
    img = cv2.Sobel(src, -1, 0, 1, delta=128)
    # cv2.imwrite(f'./{idx}_gray.png',img)

    resize_img = cv2.resize(img, dsize=(512,512))

    norm_x = np.linalg.norm(resize_img[0])
    norm_y = np.linalg.norm(resize_img[1])
    
    mini_arr = np.array([[norm_x, norm_y]])
    if idx == 0:
        big_arr = mini_arr
    else:
        big_arr = np.concatenate((big_arr, mini_arr), axis=0)

    if idx % 1000 == 0:
        print(idx)    
        pickle.dump(big_arr, open(f'./big_arr_{idx}','wb'))
#     plt.clf()

# print(big_arr.shape)
# pickle.dump(big_arr, open(f'./big_arr','wb'))

big_arr = pickle.load(open(f'./big_arr_3000','rb'))
big_arr = minmax_scale(big_arr)
df = pd.DataFrame(big_arr, columns=['x', 'y'])
estimator = KMeans(n_clusters = 5)
cluster_ids = estimator.fit_predict(df[['x', 'y']])

plt.scatter(df['x'], df['y'], c=cluster_ids)

for index, x, y in df.itertuples():
    plt.annotate(index,(x, y), fontsize=5)

plt.show()
plt.savefig('./final.png')