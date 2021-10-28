import os 
import random
from lmdb_helper import MyLMDB

def load_class_list(class_txt_path):
    with open(class_txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()    
    char_class_list = []
    for line in lines:
        # line = line.replace(' ','').replace('\n','')
        line = line.split('\t')[0]
        char_class_list.append(line)
    char_extend_list = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890()')
    char_class_list.extend(char_extend_list)
    return char_class_list

def is_in_class_list(label, char_class_list):
    for char in label:
        if not char in char_class_list:
            return False
    return True


def merge_db(root_path, src_db_path, target_path):

    target_db = MyLMDB(target_path, mode='w', map_size=60e9)

    for db_name in src_db_path:
        db_path = os.path.join(root_path, db_name)
        src_db = MyLMDB(db_path, mode='r')
        num_of_samples = src_db.num_of_samples
        char_class_list = load_class_list('/home/gucheol/다운로드/naver_recog/kr_labels.txt')
        for i in range(1, num_of_samples):
            im, label = src_db.read_image_label(i)
            
            # if len(label) > 25 or ' ' in label:
            #     print(f'skip {label}')
            #     continue
            # if not is_in_class_list(label, char_class_list):
            #     print(f'skip {label}')
            #     continue
                
            target_db.write_im_label(im, label)
    target_db.close()       


if __name__== '__main__':
    root = '/home/gucheol/다운로드/naver_recog/1005_gyucheol_work'
    src = ['gc_work_lmdb_test','gc_work_lmdb_train','gc_work_lmdb_val']
    target = '/home/gucheol/다운로드/naver_recog/1005_gyucheol_work/gc_work_lmdb'
    merge_db(root, src, target)
