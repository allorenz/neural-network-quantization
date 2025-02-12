import urllib.request
import os
import pathlib
import zipfile
import json
import random
from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub

with open('config.json', 'r') as file:
    config = json.load(file)

model_dir = config['model_dir']
model_path = config['model_path']
model_url = config['model_url']
model_quantized_path = config["model_quantized_path"]
coco_val_url = config['coco_val_url']
coco_annotations_url = config['coco_annotations_url']
coco_folder = config['coco_folder']
annotation_file_path = config['annotation_file_path']
images_file_path = config['images_file_path']
images_used_path = config['images_used_path']

urls = [coco_val_url, coco_annotations_url]


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def load_model():
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(model_path):
        print("OD model not found. Downloading...")
        os.makedirs('../data', exist_ok=True)
        model = hub.load(model_url)
        tf.saved_model.save(model, model_path)
        print("Download completed.")
        return model
    else:
        print("OD model already present locally.")
        model = tf.saved_model.load(model_path)
        return model
     


def load_coco_2014_dataset():
    os.makedirs(coco_folder, exist_ok=True)

    for url in urls:
        file_path = os.path.join(coco_folder, os.path.basename(url))
        extract_folder = os.path.join(coco_folder, os.path.basename(file_path).replace('.zip', ''))

        if not os.path.exists(file_path):
            print(f"Downloading {file_path}...")
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=file_path) as bar:
                urllib.request.urlretrieve(url, file_path, reporthook=bar.update_to)
            print(f"Downloaded {file_path}.")
        else:
            print(f"{file_path} already exists. Skipping download.")

        if file_path.endswith('.zip') and not os.path.exists(extract_folder):
            print(f"Extracting {file_path} to {extract_folder}...")
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_folder)
            print(f"Extraction completed for {file_path}.")
        else:
            print(f"{extract_folder} already exists. Skipping extraction.")

    print("COCO 2014 dataset download and extraction complete.")

    print("Randomly selecting 1500 pictures ..")
    images = select_random_pictures(images_file_path)
    print("Done!")
    return images



def get_image_ids():
    with open(annotation_file_path, 'r') as f:
        coco_data = json.load(f)
    image_id_to_filename = {image['file_name']: image["id"] for image in coco_data['images']}
    return image_id_to_filename


def select_random_pictures(folder_path, num_pictures=1500):
    all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    image_files = [f for f in all_files if f.lower().endswith('.jpg')]

    selected_files = random.sample(image_files, num_pictures)

    return selected_files