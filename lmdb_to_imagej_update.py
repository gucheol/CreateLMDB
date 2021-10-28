from lmdb_helper import MyLMDB
from preprocess_annotation_file import read_ocr_annotation_file, extract_patch_images
import cv2, os, re, tqdm



def find_and_revise_line(lines, origin_label, lmdb_label, imagej_path, row_num):
    count = []
    if origin_label == lmdb_label:
        return None 

    for index, line in enumerate(lines):
        split_line = line.split('\t')
        label = split_line[0]

        if label != origin_label:
            continue

        
        count.append(index)
        #     print(f"{imagej_path} should be check, {origin_label}\t{lmdb_label}")
        #     return None
    
    if len(count) == 0 :
        print(f"{imagej_path} should be check, {origin_label} is not exists")
        return None
    elif len(count) == 1:     
        index = count[0]
        split_line = lines[index].split('\t')
        split_line[0] = lmdb_label
        lines[index] = '\t'.join(split_line)
        return lines
    else:
        for index in count:
            if index == row_num:
                split_line = lines[index].split('\t')
                split_line[0] = lmdb_label
                lines[index] = '\t'.join(split_line)
                return lines
        print(f"{imagej_path} should be check, {origin_label}\t{lmdb_label}")
        return None 
    

def rewrite_txt(imagej_path, origin_label, lmdb_label, row_num):
    # for txt_file in os.listdir(imagej_path):
        # if not '.txt' in txt_file:
        #     continue
    extention = imagej_path.split('.')[-1]
    txt_path = imagej_path.replace(extention, 'txt')
    with open(txt_path,'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    revise_lines = find_and_revise_line(lines, origin_label, lmdb_label, imagej_path, row_num)
    
    if revise_lines is not None:
        with open(txt_path,'w', encoding='utf-8') as f:
            f.writelines(revise_lines)


def revise_imagej(origin_lmdb_path, revise_lmdb_path, imagej_path):
    origin_lmdb = MyLMDB(origin_lmdb_path, mode='r')
    revise_lmdb = MyLMDB(revise_lmdb_path, mode='r')

    for index in range(revise_lmdb.get_number_of_samples()):
        _, lmdb_label, lmdb_path = revise_lmdb.read_image_label_path_key(index+1)
        _, origin_label, _ = origin_lmdb.read_image_label_path(index+1)

        if lmdb_label != origin_label:
            row_num  = origin_lmdb.get_row_num(index)
            imagej_image_path = os.path.join(imagej_path,lmdb_path.split('/')[-1])
            rewrite_txt(imagej_image_path, origin_label, lmdb_label, row_num)

    # for index in range(revise_lmdb.get_number_of_samples()):
    #     _, lmdb_label, lmdb_path, lmdb_id = revise_lmdb.read_image_label_path_key(index+1)
    #     _, origin_label, _ = origin_lmdb.read_image_label_path(int(lmdb_id))
    #     row_num  = origin_lmdb.get_row_num(int(lmdb_id))
    #     imagej_image_path = os.path.join(imagej_path,lmdb_path.split('/')[-1])
    #     # if imagej_image_path.find('54456063') >= 0:            
    #     rewrite_txt(imagej_image_path, origin_label, lmdb_label, row_num)

###########  수정필요한 이미지 파일의 id 데이터를 넣어 생성하는 lmdb ##########
def write_lmdb_with_id(read_lmdb_path, write_lmdb_path, image_folder):
    r_lmdb = MyLMDB(read_lmdb_path, mode='r', sync_period=1000)
    w_lmdb = MyLMDB(write_lmdb_path, mode='w')
    id_list = [file_name.split('_')[0] for file_name in os.listdir(image_folder) if '.jpg' in file_name]
    
    for lmdb_id in id_list:
        im, label, path = r_lmdb.read_image_label_path(int(lmdb_id))
        w_lmdb.write_im_label_path(im,label,path,lmdb_id)
    w_lmdb.close()


if __name__== '__main__':
    # r_lmdb_path = '/home/gucheol/다운로드/naver_recog/0929_lmdb'
    # w_lmdb_path = '/home/gucheol/다운로드/naver_recog/0929_result_gu_part2_lmdb'
    # image_folder = '/home/gucheol/다운로드/naver_recog/0929_result_gu_part2/수정필요'
    # write_lmdb_with_id(r_lmdb_path, w_lmdb_path, image_folder)

    origin_lmdb_path = '/home/gucheol/다운로드/general_app_val_lmdb'
    revise_lmdb_path = '/home/gucheol/다운로드/general_app_val_lmdb_revised'
    imagej_path = '/home/gucheol/다운로드/general_app_val_original'
    revise_imagej(origin_lmdb_path, revise_lmdb_path, imagej_path)