import os 
from lmdb_helper import MyLMDB


def count_char(target_db, output_path=''):
    num_samples = target_db.get_number_of_samples()
    char_dict = {}
    max_len = 0
    for index in range(1, num_samples+1):
        _, label = target_db.read_image_label(index)
        if max_len < len(label):
            max_len = len(label)
        char_list = [char for char in label]
        for char in char_list:
            if char not in char_dict:
                char_dict[char] = 1
            else:
                char_dict[char] += 1
    return char_dict


def convert_string_format(sorted_dict):
    text_list =[]
    for key,val in sorted_dict:
        line = str(key) +'\t'+ str(val) + '\n'
        text_list.append(line)
    return text_list


def load_class_list(class_txt_path):
    with open(class_txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()    
    char_class_list = []
    for line in lines:
        line = line.replace(' ','').replace('\n','')
        char_class_list.append(line)
    return char_class_list


def char_num_in_class_list(char_dict, class_info_list, output_path=''):
    removed_not_in_class = {}
    for char in class_info_list:
        if char in list(char_dict.keys()):
            removed_not_in_class[char] = char_dict[char]

    sorted_dict = sorted(removed_not_in_class.items(),key=lambda x: x[1])
    text_list = convert_string_format(sorted_dict)
    if not output_path == '':
        out_path = f'{output_path}/removed_class_num.txt'
        with open(out_path, 'w', encoding='utf8') as f:
            f.writelines(text_list)
    return removed_not_in_class


def check_char(char_dict_list, class_info_list, output_path =''):
    class_dict = {}
    for char in class_info_list:
        class_dict[char] = 0
    
    for char_dict in char_dict_list:
        for char in list(char_dict.keys()):
            if char not in list(class_dict.keys()):
                print(f'{char} is not in class_list')
            else:
                class_dict[char] = class_dict[char] + char_dict[char]
    
    if not output_path == '':
        sorted_dict = sorted(class_dict.items(),key=lambda x: x[1])
        text_list = convert_string_format(sorted_dict)
        out_path = os.path.join(output_path,'lmdb_char_class_num.txt')
        with open(out_path, 'w', encoding='utf8') as f:
            f.writelines(text_list)


if __name__== '__main__':
    class_info_txt_path = '/home/gucheol/data/tmp/ko_char.txt'
    data_dir = '/home/gucheol/data/hc_recog_data/val'
    lmdb_target = ['hc_real_lmdb']
    result_txt_path = '/home/gucheol/data/tmp'

    # load lmdb
    char_dict_list =[]
    for lmdb_name in lmdb_target:
        lmdb_path = os.path.join(data_dir,lmdb_name)
        target_db = MyLMDB(lmdb_path, mode='r')
        char_dict = count_char(target_db, output_path=result_txt_path)
        char_dict_list.append(char_dict)
    
    # class info 
    class_info_list = load_class_list(class_info_txt_path)
    char_extend_list = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890()')
    class_info_list.extend(char_extend_list)

    # check data
    check_char(char_dict_list, class_info_list,'/home/gucheol/data/hc_recog_data/train')


    # char_num_in_class_list(char_dict, class_info_list, output_path=result_txt_path)
    # print(set(class_info_list) - set(char_dict.keys()))
        
