from data_model_loader import load_coco_2014_dataset
from quantization import quantize_models


def main():
    images = load_coco_2014_dataset()
    #model = load_model()
    quantize_models()
    

if __name__ == '__main__':
    main()