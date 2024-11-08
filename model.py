from data_model_loader import *
from PIL import Image
import torchvision.transforms as transforms
import pathlib
from tqdm import tqdm

with open('config.json', 'r') as file:
    config = json.load(file)

coco_folder = config['coco_folder']
annotation_file_path = config['annotation_file_path']


def predict(model, data_path, images, n_images=5):
    data_path = pathlib.Path(data_path)
    predictions = {}

    for img in tqdm(images[:n_images], desc="Loading images"):
        # load image
        image_path = data_path/"val2014"/"val2014"/img
        image = Image.open(image_path)

        # transform to tensor
        transform = transforms.Compose([
            transforms.ToTensor()  # Konvertiert das Bild zu einem Tensor [C, H, W] mit Werten zwischen 0 und 1
        ])
        image_tensor = transform(image)
        image_tensor = (image_tensor * 255).byte()  # convert to uint8
        image_tensor = image_tensor.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]
        image_tensor = image_tensor.unsqueeze(0)

        # inference
        detector_output = model(image_tensor)
        score = detector_output["detection_scores"].numpy()[0]
        detected_class = detector_output["detection_classes"].numpy()[0]
        
        # eval
        predictions[img] = {
            "detected_classes": detected_class,
            "scores": score
        }
        
    return predictions