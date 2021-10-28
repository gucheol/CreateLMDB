import cv2
import os
import glob
import numpy as np
import logging
logging.getLogger().setLevel(logging.INFO)


# def read_segmentation_annotation_file(path, definitions):
#     annotation = {}
#     with open(path, 'r') as f:
#         lines = f.readlines()
#     for line in lines:
#         items = line.strip().split('\t')
#         label = items[0]
#         if definitions.get(label) is None:
#             print(f'{label} undefined label name ')
#             exit(1)
#         points = np.array([max(float(x), 0) for x in items[1:9]], dtype=np.int32).reshape(-1, 2)
#         annotation[label] = points
#     return annotation


# def annotation_to_segmentation(image, annotation, definition):
#     segmentation = np.zeros_like(image)
#     # 그냥 fill 하면 background 가 Person Image 를 덮게 된다.
#     # 꼼수이기는 하지만 annotation items 을 label 기준으로 sorting 해서
#     # Background 가 항상 처음에 나오도록 한다.
#     for label, points in sorted(annotation.items()):
#         rgb = definition[label]
#         cv2.fillPoly(segmentation, [points], tuple(rgb.tolist()), None)
#     return segmentation


# def make_mask_file_from_annotation(annotation_path, target_path, definition):
#     for file_path in glob.glob(annotation_path + "*.jpg"):
#         im = cv2.imread(file_path)
#         annotation_file_path = file_path.replace(".jpg", ".txt")
#         segmentation_annotation = read_segmentation_annotation_file(annotation_file_path, definition)
#         segmentation_im = annotation_to_segmentation(im, segmentation_annotation, definition)
#         cv2.imwrite(os.path.join(target_path, os.path.basename(file_path)), segmentation_im)


# def create_image_patch_by_segmentation_annotation(annotation_path, target_path, definition):
#     for file_path in glob.glob(annotation_path + "*.jpg"):
#         im = cv2.imread(file_path)
#         annotation_file_path = file_path.replace(".jpg", ".txt")
#         segmentation_annotation = read_segmentation_annotation_file(annotation_file_path, definition)
#         patch_images = extract_patch_images(im, segmentation_annotation)
#         patch_name = os.path.join(target_path, os.path.basename(file_path).replace('.txt', ''))
#         for label, image in patch_images.items():
#             patch_path = patch_name + f'_{label}.png'
#             cv2.imwrite(patch_path, image)


# def create_image_patch_by_ocr_annotation(annotation_path, target_path, without_label=False):
#     for (root, dirs, file_names) in os.walk(annotation_path):
#         for file_name in file_names:
#             file_path = os.path.join(root, file_name)
#             if file_path.split('.')[-1] != 'jpg':
#                 continue

#             print(f'{file_path} is processing')
#             im = cv2.imread(file_path)
#             annotation_file_path = file_path.replace(".jpg", ".txt")
#             if not os.path.exists(annotation_file_path):
#                 print(f'error {annotation_file_path} did not exist')
#                 continue
#             ocr_annotation = read_ocr_annotation_file(annotation_file_path)            
#             patch_images = extract_patch_images(im, ocr_annotation)
#             patch_name = os.path.join(target_path, os.path.basename(file_path).replace('.jpg', ''))
#             for index, (label, image) in enumerate(patch_images.items()):
#                 if without_label:
#                     patch_path = patch_name + f'_L__L_{index}.png'    
#                 else:
#                     patch_path = patch_name + f'_L_{label}_L_{index}.png'

#                 try:
#                     print(f'{patch_path}')
#                     cv2.imwrite(patch_path, image)
#                 except:
#                     print(f'error {patch_path} has empty image for {file_path}')


# def create_image_patch_by_ocr_annotation_file(file_path, target_path):
#     im = cv2.imread(file_path)
#     annotation_file_path = file_path.replace(".jpg", ".txt")
#     if not os.path.exists(annotation_file_path):
#         print(f'error {annotation_file_path} did not exist')
#         return None

#     ocr_annotation = read_ocr_annotation_file(annotation_file_path)
#     patch_images = extract_patch_images(im, ocr_annotation)
#     patch_name = os.path.join(target_path, os.path.basename(file_path).replace('.jpg', ''))
#     for index, (label, image) in enumerate(patch_images.items()):
#         patch_path = patch_name + f'_L_{label}_L_{index}.png'
#         try:
#             cv2.imwrite(patch_path, image)
#         except:
#             print(f'error {patch_path} has empty image for {file_path}')

############ imageJ ocr ################
def create_LMDB_by_ocr_annotation(annotation_path, target_path, divide_ratio={"train": 7, "val": 2, "test": 1}):
    from lmdb_helper import MyLMDB
    from random import sample, random
    
    lmdb_list = create_LMDBs(divide_ratio)
    # *이나 #이 포함된 라벨만 따로 모아 놓음
    lmdb_ignore = MyLMDB(f'{target_path}_ignore', sync_period=1000) 


    for (root, dirs, file_names) in os.walk(annotation_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            extension = file_path.split('.')[-1]

            if not is_img_file(extension):
                continue

            logging.info(f'{file_path} is processing')

            im = read_img(file_path)
            annotation_file_path = get_annotation_file_path(file_path, extension)
            ocr_annotation = read_ocr_annotation_file(annotation_file_path)            
            patch_images = extract_patch_images(im, ocr_annotation)

            # 패치를 하나씩 읽어, 각 lmdb에 데이터 쓰기
            for index, (label, im) in enumerate(patch_images.items()):                
                if label.find("#") >= 0 or label.find("*") >= 0:
                    lmdb_ignore.write_im_label_path(im, label, file_path, str(index))
                    continue

                h, w = im.shape[:2]
                if h > w*1.5 and len(label) > 1:
                    im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
                
                ratios = [divide_ratio[x] for x in divide_ratio]
                db_index = choose_lmdb(random(), ratios, len(divide_ratio))
                lmdb_list[db_index].write_im_label_path(im, label, file_path, str(index))
       
    for lmdb in lmdb_list:
        lmdb.close()
    lmdb_ignore.close()


def create_LMDBs(divide_ratio):
    from lmdb_helper import MyLMDB

    lmdb_list = []
    for _, (db_type, ratio) in enumerate(divide_ratio.items()):
        if not type(ratio) == int:
            logging.error("divide_ratio 값이 int가 아닙니다.")
            exit(1)
        lmdb_list.append(MyLMDB(f'{target_path}_{db_type}', sync_period=1000))
        logging.info(f"{db_type}이 비율 {ratio}로 생성됩니다.")
    return lmdb_list


def is_img_file(extension):
    if extension not in ['jpg','jpeg']:
        return False
    return True


def read_img(file_path):
    im = cv2.imread(file_path)
    if im is None:
        logging.error(f'can not read image: {file_path}')
        exit (-1)
    return im


def get_annotation_file_path(file_path, extension):
    annotation_file_path = file_path.replace(extension, "txt")
    if not os.path.exists(annotation_file_path):
        logging.error(f'error {annotation_file_path} did not exist')
        exit (-1)
    return annotation_file_path


def read_ocr_annotation_file(path):
    annotation = {}
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        items = line.strip().split('\t')
        label = items[0]
        try:
            points = np.array([max(float(x), 0) for x in items[1:9]], dtype=np.int32).reshape(-1, 2)
            points = align_points(points)
        except:
            print(f'error {path} {str(items)}')
        annotation[label] = points
    return annotation


def align_points(box):
    # Make it clockwise align
    # box = np.array([(int(float(x)), int(float(y))) for x, y in zip(items[1::2], items[2::2])])
    centroid = np.sum(box, axis=0) / 4
    theta = np.arctan2(box[:, 1] - centroid[1], box[:, 0] - centroid[0]) * 180 / np.pi
    indices = np.argsort(theta)
    aligned_box = box[indices]
    return aligned_box


def extract_patch_images(im, annotation, method='WARP_TRANSFORM'):
    # point 는 x, y 순서로 들어온다.
    patch_images = {}
    for label, points in annotation.items():
        # 첫 번째 방법 points 를 모두 포함하는 bounding box 크기의 이미지를 흰색 배경으로 만들고 paste
        # 두 번째 방법 points 가 사각일 때 perspective warp 수행해서 만들기
        # 일단 첫 번째 방법 부터 해보자.
        # if label.find("<") == -1:
        #     continue
        if method == 'SIMPLE':
            x, y, width, height = cv2.boundingRect(points)
            patch_images[label] = im[y: y + height, x: x + width]
        elif method == 'MASKING': # masking으로 하면 테두리가 생기고 배경이 완전 흰색이 아니기 때문에 여전히 이질적이다.
            mask_file = np.zeros_like(im)
            fg_mask = cv2.fillPoly(mask_file, [points], (255, 255, 255), None)
            contents_im = cv2.bitwise_and(im, fg_mask)
            bg_mask = cv2.bitwise_not(fg_mask)
            target_im = cv2.bitwise_or(contents_im, bg_mask, fg_mask)
            x, y, width, height = cv2.boundingRect(points)
            patch_images[label] = target_im[y: y + height, x: x + width]
        elif method == 'WARP_TRANSFORM':
            patch_images[label] = four_point_transform(im, points.astype(np.float32))
    return patch_images


def four_point_transform(image, rect):
    (tl, tr, br, bl) = rect
    if tl[0] < tr[0] and tl[1] < bl[1] and tr[1] < br[1] and bl[0] < br[0]:
        pass
    else:
        print('error')
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1],[0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    cv2.imwrite('wapred.png', warped)
    return warped


def choose_lmdb(ran_num, ratios, num):
    """
    ran_num : 0~1 사이의 랜덤한 숫자
    devide_ratio : 데이터를 나눌 비율, 이것으로 확률의 Treshold를 계산. ex) [7,2,1]
    num : devide_ratio의 인덱스 갯수
    return : devide_ratio 확률에 기반한, devide_ratio의 index
    설명 : 재귀함수를 이용한, 리스트 안의 숫자의 크기를 참조하여, 인덱스를 리턴
    """
    if num < 0:
        return

    cur_index = len(ratios) - num
    threshold = np.sum([ratios[x] for x in range(0, cur_index+1)])/np.sum(ratios)
    if ran_num < threshold:
        return cur_index
    else:
        num -= 1
        return choose_lmdb(ran_num, ratios, num)


if __name__ == '__main__':
    # source_path = './data/passport/'
    # destination_path = './result/mask'
    # definition_path = './data/Train/annotation_definitions.json'
    # segmentation_definition = load_segmentation_definition(definition_path)
    # make_mask_file_from_annotation(source_path, destination_path, segmentation_definition)
    # # create_image_patch_by_segmentation_annotation(source_path, destination_path, segmentation_definition)
    # create_image_patch_by_ocr_annotation(source_path, destination_path)
    # # create_image_patch_by_ocr_annotation_file('./data/passport/스리랑카(8)/0_0_147234.jpg', './result/mask/')

    # source_path = '../../data/nullee_invoice/TR_data/'
    # destination_path = '../../data/nullee_invoice/TR_data_patch/'
    # if not os.path.exists(destination_path):
    #     os.mkdir(destination_path)

    # for folder_name in os.listdir(source_path):
    #     data_path = os.path.join(source_path, folder_name)
    #     target_path = os.path.join(destination_path, folder_name)
    #     if not os.path.exists(target_path):
    #         os.mkdir(target_path)        
    #     create_image_patch_by_ocr_annotation(data_path, target_path, without_label=True)

    # file_path = '../../data/nullee_invoice/TR_data/계산서/DEV00000000015634386590710000.jpg'
    # im = cv2.imread(file_path)
    # annotation_file_path = file_path.replace('.jpg', '.txt')
    # ocr_annotation = read_ocr_annotation_file(annotation_file_path)            
    # patch_images = extract_patch_images(im, ocr_annotation)
    

    source_path = '/home/gucheol/다운로드/test'
    target_path = '/home/gucheol/다운로드/test_lmdb'
    create_LMDB_by_ocr_annotation(source_path, target_path, divide_ratio={"train": 7, "val": 2, "test": 1})