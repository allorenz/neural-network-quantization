from data_model_loader import *
from PIL import Image
import torchvision.transforms as transforms
import pathlib
from tqdm import tqdm
import time
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


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


def predict(model, images, n_images=5, data_path=COCO_FOLDER):
    data_path = pathlib.Path(data_path)
    results = []
    inference_times = []

    for img in tqdm(images[:n_images], desc="Inferencing images"):
        # load image
        image_path = data_path/"val2014"/"val2014"/img
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
                    "image_id" : int(filename_to_image_id[img]),
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

def evaluate_predictions(results_dir=RESULTS_DIR, model_name=MODEL_NAME):
    annType = 'bbox'
    results_file_path = results_dir + model_name + "_results.json"

    #initialize COCO ground truth api
    cocoGt=COCO(ANNOTATION_FILE_PATH)

    #initialize COCO detections api
    cocoDt=cocoGt.loadRes(results_file_path)

    # prepare ids
    imgIds=sorted(cocoGt.getImgIds())

    # evaluate
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    # average precision scores
    #ap_scores = cocoEval.stats
    metrics = generate_metrics_dict(cocoEval)

    return metrics

def generate_metrics_dict(cocoEval):
    keys = [
        "AP_IoU_0.50:0.95_all_maxDets_100",
        "AP_IoU_0.50_all_maxDets_100",
        "AP_IoU_0.75_all_maxDets_100",
        "AP_IoU_0.50:0.95_small_maxDets_100",
        "AP_IoU_0.50:0.95_medium_maxDets_100",
        "AP_IoU_0.50:0.95_large_maxDets_100",
        "AR_IoU_0.50:0.95_all_maxDets_1",
        "AR_IoU_0.50:0.95_all_maxDets_10",
        "AR_IoU_0.50:0.95_all_maxDets_100",
        "AR_IoU_0.50:0.95_small_maxDets_100",
        "AR_IoU_0.50:0.95_medium_maxDets_100",
        "AR_IoU_0.50:0.95_large_maxDets_100"
    ]
    metrics = {key: cocoEval.stats[i] for i, key in enumerate(keys)}
    return metrics