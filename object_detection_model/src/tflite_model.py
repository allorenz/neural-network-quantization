import tensorflow as tf
import pathlib
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image
import json
import time


# load file paths
with open('config.json', 'r') as file:
    config = json.load(file)
ANNOTATION_FILE_PATH = config['annotation_file_path']
RESULTS_DIR = '../results/'
COCO_FOLDER = config['coco_folder']

# load image ids
with open(ANNOTATION_FILE_PATH, 'r') as f:
        coco_data = json.load(f)
filename_to_image_id= {image['file_name']: image["id"] for image in coco_data['images']}


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


def predict(interpreter, image_names, n_images=None, data_path=COCO_FOLDER):
    data_path = pathlib.Path(data_path)
    results = []
    inference_times = []
    # load model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # create name mapping for output object
    name_map = {value["name"] : output_name for output_name, value in interpreter.get_signature_runner().get_output_details().items()}

    for image_name in tqdm(image_names[:n_images], desc="Inferencing images"):
        # load and prepare image
        image_path = data_path/"val2014"/"val2014"/image_name
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
                    "image_id" : int(filename_to_image_id[image_name]),
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