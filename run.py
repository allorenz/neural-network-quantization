from data_model_loader import *
def main():
    model = load_yolov10_model()
    annotations, images = load_coco_2014_dataset()


if __name__ == '__main__':
    main()