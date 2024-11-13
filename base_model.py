import json
from PIL import Image
import torchvision.transforms as transforms
import pathlib
from tqdm import tqdm
import time


# load file paths
with open('config.json', 'r') as file:
    config = json.load(file)
COCO_FOLDER = config['coco_folder']
ANNOTATION_FILE_PATH = config['annotation_file_path']
RESULTS_DIR = 'results/'
MODEL_NAME = "ssd_mobilenet"

# load image ids
with open(ANNOTATION_FILE_PATH, 'r') as f:
        coco_data = json.load(f)
filename_to_image_id= {image['file_name']: image["id"] for image in coco_data['images']}


def predict(model, image_names, n_images=None, data_path=COCO_FOLDER):
    data_path = pathlib.Path(data_path)
    results = []
    inference_times = []

    for image_name in tqdm(image_names[:n_images], desc="Inferencing images"):
        # load image
        image_path = data_path/"val2014"/"val2014"/image_name
        image = Image.open(image_path)
        width, height = image.size

        # transform
        image_tensor = preprocess_image(image)

        # inference
        start = time.time()
        detector_output = model(image_tensor)
        end = time.time()
              
        # evaluate
        inference_times.append(end - start)
        n_detections = int(detector_output["num_detections"].numpy()[0])

        for i in range(n_detections):
            ymin, xmin, ymax, xmax = detector_output["detection_boxes"].numpy()[0][i]
            ymin = ymin * height
            ymax = ymax * height
            xmin = xmin * width
            xmax = xmax * width

            result = {
                    "image_id" : int(filename_to_image_id[image_name]),
                    "category_id":int(detector_output["detection_classes"].numpy()[0][i]),
                    "bbox": [xmin, ymin, xmax - xmin, ymax - ymin], # detector_output["detection_boxes"].numpy()[0][i].tolist(), # needs to be list
                    "score": float(detector_output["detection_scores"].numpy()[0][i])
            }
            results.append(result)
    
    return results, inference_times

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

def store_results(results, results_dir):
    # create dir
    results_dir = pathlib.Path(results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    results_file = results_dir/f"{MODEL_NAME}_results.json"

    # store results
    with open(results_file, 'w') as json_file:
        json.dump(results, json_file, indent=4)