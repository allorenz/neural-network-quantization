from data_model_loader import *
from PIL import Image
import torchvision.transforms as transforms
import pathlib
from tqdm import tqdm


# load file paths
with open('config.json', 'r') as file:
    config = json.load(file)
coco_folder = config['coco_folder']
annotation_file_path = config['annotation_file_path']


# load image ids
with open(annotation_file_path, 'r') as f:
        coco_data = json.load(f)
filename_to_image_id= {image['file_name']: image["id"] for image in coco_data['images']}


def predict(model, data_path, images, n_images=5):
    data_path = pathlib.Path(data_path)
    results = []

    for img in tqdm(images[:n_images], desc="Inferencing images"):
        # load image
        image_path = data_path/"val2014"/"val2014"/img
        image = Image.open(image_path)
        width, height = image.size

        # transform
        image_tensor = preprocess_image(image)

        # inference
        detector_output = model(image_tensor)
              
        # evaluate
        n_detections = int(detector_output["num_detections"].numpy()[0])
        #print(detector_output["detection_boxes"])
        for i in range(n_detections):
            ymin, xmin, ymax, xmax = detector_output["detection_boxes"].numpy()[0][i]
            ymin = ymin * height
            ymax = ymax * height
            xmin = xmin * width
            xmax = xmax * width

            result = {
                    "image_id" : int(filename_to_image_id[img]),
                    "category_id":int(detector_output["detection_classes"].numpy()[0][i]),
                    "bbox": [xmin, ymin, xmax - xmin, ymax - ymin], # detector_output["detection_boxes"].numpy()[0][i].tolist(), # needs to be list
                    "score": float(detector_output["detection_scores"].numpy()[0][i])
            }
            results.append(result)
    
    return results


def preprocess_image(image):
    # convert greyscale to rgb
    if image.mode == 'L':
        image = image.convert('RGB')

    transform = transforms.Compose([
        transforms.ToTensor()  # Converts the image to a tensor [C, H, W] with values between 0 and 1
    ])
    
    image_tensor = transform(image)
    image_tensor = (image_tensor * 255).byte()
    image_tensor = image_tensor.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]   
    # Add a batch dimension [1, H, W, C]
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor
