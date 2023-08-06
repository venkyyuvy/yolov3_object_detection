# yolov3_object_detection
Training the YOLO V3 object detection model from scratch on the Pascal VOC dataset

## Types of object detection:

- Basic object detection: Detecting the four corners of the rectangle cover the object.
- Semantic segmentation: pixel wise identifying the contours of different objects.
- Instance segmentation: pixel wise identifying the contours of every instance of different objects.
- panoptic segmentation: Every pixel is labelled into two categories background and things (person, vehicles, etc.).


## Two types of object detection:

- One stage networks: Single network directly predicts the bounding box of the objects from grid (s*s grid from the image). Examples: YOLO, SSD, Retina-net, EfficientDet.

- Two stage networks: One network (or other methods) takes care of the region proposal.