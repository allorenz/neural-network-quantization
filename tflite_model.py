import tensorflow as tf
import pathlib
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image
import json
import time
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# load file paths
with open('config.json', 'r') as file:
    config = json.load(file)
ANNOTATION_FILE_PATH = config['annotation_file_path']
RESULTS_DIR = 'results/'

# load image ids
with open(ANNOTATION_FILE_PATH, 'r') as f:
        coco_data = json.load(f)
filename_to_image_id= {image['file_name']: image["id"] for image in coco_data['images']}


def load_tflite_model(tflite_model_path):
    # load tflite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    # get input and output tensors of model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    return interpreter, input_details, output_details

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


def predict(tflite_model_path, data_path, images, n_images=5):

    data_path = pathlib.Path(data_path)
    results = []
    inference_times = []

    # load model
    interpreter, input_details, output_details = load_tflite_model(tflite_model_path)
    
    # create name mapping for output object
    name_map = {value["name"] : output_name for output_name, value in interpreter.get_signature_runner().get_output_details().items()}

    for img in tqdm(images[:n_images], desc="Inferencing images"):
        # load and prepare image
        image_path = data_path/"val2014"/"val2014"/img
        image = Image.open(image_path)
        width, height = image.size
        
        # transform image
        image_tensor = preprocess_image(image)

        # resize model input for image size
        interpreter.resize_tensor_input(input_details[0]['index'], image_tensor.size())
        interpreter.allocate_tensors()

        # pass input image tensor for inference
        interpreter.set_tensor(input_details[0]['index'], image_tensor)

        # inference - results are automatically stored in "output_details"
        start = time.time()
        interpreter.invoke()
        end = time.time()
        inference_times.append(end - start)

        # prepare output object
        detector_output = {name_map[output_detail['name']]:interpreter.get_tensor(output_detail['index']) for output_detail in output_details}
        
        # evaluate
        n_detections = int(detector_output["num_detections"][0])
        for i in range(n_detections):
            ymin, xmin, ymax, xmax = detector_output["detection_boxes"][0][i]
            ymin = ymin * height
            ymax = ymax * height
            xmin = xmin * width
            xmax = xmax * width

            result = {
                    "image_id" : int(filename_to_image_id[img]),
                    "category_id":int(detector_output["detection_classes"][0][i]),
                    "bbox": [xmin, ymin, xmax - xmin, ymax - ymin], # detector_output["detection_boxes"].numpy()[0][i].tolist(), # needs to be list
                    "score": float(detector_output["detection_scores"][0][i])
            }

            results.append(result)
    
    return results, inference_times

def store_results(results, results_dir, model_name):

    # create dir
    results_dir = pathlib.Path(results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    results_file = results_dir/f"{model_name}_results.json"

    # store results
    with open(results_file, 'w') as json_file:
        json.dump(results, json_file, indent=4)

def evaluate_predictions(model_name, results_dir=RESULTS_DIR):
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