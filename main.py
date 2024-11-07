from data_model_loader import *


def main():
    annotations, images = load_coco_2014_dataset()
    model = load_model()
    quantize_models()
    

if __name__ == '__main__':
    main()