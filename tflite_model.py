import tensorflow as tf
import pathlib
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image
import json


# load file paths
with open('config.json', 'r') as file:
    config = json.load(file)
annotation_file_path = config['annotation_file_path']

# load image ids
with open(annotation_file_path, 'r') as f:
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


    for img in tqdm(images[:n_images], desc="Inferencing images"):
        # load and prepare image
        image_path = data_path/"val2014"/"val2014"/img
        image = Image.open(image_path)
        width, height = image.size
        
        # transform image
        image_tensor = preprocess_image(image)

        # load model, resize model input
        interpreter, input_details, output_details = load_tflite_model(tflite_model_path)
        interpreter.resize_tensor_input(input_details[0]['index'], image_tensor.size())
        interpreter.allocate_tensors()

        # pass input image tensor for inference
        interpreter.set_tensor(input_details[0]['index'], image_tensor)

        # inference - results are automatically stored in "output_details"
        interpreter.invoke()

        # create name mapping
        name_map = {value["name"] : output_name for output_name, value in interpreter.get_signature_runner().get_output_details().items()}
        
        # prepare output object
        detector_output = {}
        for output_detail in output_details:
            detector_output[name_map[output_detail['name']]] = interpreter.get_tensor(output_detail['index'])

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
    
    return results