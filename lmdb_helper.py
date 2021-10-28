import os
import io
import lmdb
import six
import cv2
from PIL import Image

IMAGE_SAMPLE_HEIGHT = 64

def image_bin_to_pil(image_bin):
    buf = six.BytesIO()
    buf.write(image_bin)
    buf.seek(0)
    img = Image.open(buf)
    return img
 

def is_valid_label(label, classes):
    for ch in label:
        if classes.get(ch) is None:
            print(f'{ch} is not valid')
            return False
    return True


def load_class_dictionary(path, add_space=False):
    class_dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        items = line.strip().split()
        # class_dict[items[1]] = items[0]
        class_dict[items[0]] = items[1]
    return class_dict


def load_and_resize(path, label, resize=False):
    im = Image.open(path)

    w, h = im.size
    if h > w * 1.5 and len(label) > 1:
        im = im.rotate(90.0, expand=True)
    
    if resize: 
        scaled_w = int(IMAGE_SAMPLE_HEIGHT / h * w)
        im = im.resize((scaled_w, IMAGE_SAMPLE_HEIGHT), Image.LANCZOS)

    with io.BytesIO() as output:
        im.save(output, format="JPEG")
        contents = output.getvalue()
    return contents


class MyLMDB:
    def __init__(self, path, sync_period=100, mode='w', map_size=1e9):
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.session = lmdb.open(path, map_size=10e10)
        self.cache = {}
        self.sync_period = sync_period
        self.num_of_write = 0
        self.num_of_samples = self.get_number_of_samples()
        self.mode = mode 

    def get_number_of_samples(self):
        with self.session.begin(write=False) as txn:
            num_samples = txn.get('num-samples'.encode())
            if num_samples is None:
                num_of_samples = 1
            else:
                num_of_samples = int(num_samples)
        return num_of_samples

    def write_im_label_path(self, im, label, path, row_index, resize=True): 
        image_key = 'image-%09d'.encode() % self.num_of_samples
        label_key = 'label-%09d'.encode() % self.num_of_samples
        path_key = 'path-%09d'.encode() % self.num_of_samples
        row_index_key = 'row_index_key-%09d'.encode() % self.num_of_samples

        _, image_bin = cv2.imencode('.jpeg', im)
        # image_bin = im

        self.cache[image_key] = image_bin
        self.cache[label_key] = label.encode()
        self.cache[path_key] = path.encode()
        self.cache[row_index_key] = row_index.encode()

        self.num_of_samples += 1
        self.num_of_write += 1
        if self.num_of_write > self.sync_period:
            print(f'{self.path} cache write {self.num_of_samples}')
            self.cache['num-samples'.encode()] = str(self.num_of_samples - 1).encode()
            with self.session.begin(write=True) as txn:
                for k, v in self.cache.items():
                    txn.put(k, v)
            self.num_of_write = 0
            self.cache = {}


    def write_image_label(self, image_path, label, resize=True):
        image_bin = load_and_resize(image_path, label, resize)

        image_key = 'image-%09d'.encode() % self.num_of_samples
        label_key = 'label-%09d'.encode() % self.num_of_samples
        self.cache[image_key] = image_bin
        self.cache[label_key] = label.encode()

        self.num_of_samples += 1
        self.num_of_write += 1
        if self.num_of_write > self.sync_period:
            print(f'{self.path} cache write {self.num_of_samples}')
            self.cache['num-samples'.encode()] = str(self.num_of_samples - 1).encode()
            with self.session.begin(write=True) as txn:
                for k, v in self.cache.items():
                    txn.put(k, v)
            self.num_of_write = 0
            self.cache = {}

    def read_image_label(self, index):
        label_key = 'label-%09d'.encode() % index
        image_key = 'image-%09d'.encode() % index

        with self.session.begin(write=False) as txn:
            im = txn.get(image_key)
            label = txn.get(label_key).decode('utf-8')
            return im, label

    def read_image_label_path(self, index):
        label_key = 'label-%09d'.encode() % index
        image_key = 'image-%09d'.encode() % index
        path_key = 'path-%09d'.encode() % index

        with self.session.begin(write=False) as txn:
            im = txn.get(image_key)
            label = txn.get(label_key).decode('utf-8')
            path = txn.get(path_key).decode('utf-8')
            return im, label, path

    def read_image_label_path_key(self, index):
        label_key = 'label-%09d'.encode() % index
        image_key = 'image-%09d'.encode() % index
        path_key = 'path-%09d'.encode() % index
        row_index_key = 'row_index-%09d'.encode() % index

        with self.session.begin(write=False) as txn:
            im = txn.get(image_key)
            label = txn.get(label_key).decode('utf-8')
            path = txn.get(path_key).decode('utf-8')
            row_index = txn.get(row_index_key).decode('utf-8')
            return im, label, path, row_index

    def get_row_num(self, index):
        label_key = 'label-%09d'.encode() % index
        image_key = 'image-%09d'.encode() % index
        path_key = 'path-%09d'.encode() % index
        # id_key = 'id-%09d'.encode() % index

        with self.session.begin(write=False) as txn:
            im = txn.get(image_key)
            label = txn.get(label_key).decode('utf-8')
            path = txn.get(path_key).decode('utf-8')
            # id = txn.get(id_key).decode('utf-8')

            row_num = 0 
            index -= 1        
            while index >= 1:
                prev_path_key = 'path-%09d'.encode() % index
                prev_path = txn.get(prev_path_key).decode('utf-8')
                if prev_path != path:
                    break
                index -= 1
                row_num += 1
            
        return row_num
            

    def close(self):
        if self.mode != 'w':
            return 
        self.cache['num-samples'.encode()] = str(self.num_of_samples - 1).encode()
        with self.session.begin(write=True) as txn:
            for k, v in self.cache.items():
                txn.put(k, v)