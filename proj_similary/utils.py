import os

def aug_img(dir_path):
    '''
    폴더 내 모든 file list를 불러 온 뒤,
    dataFrame을 하나로 묶는 과정
    '''

    file_list = []

    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            file_list.append(dir_path + path)

    print(file_list)