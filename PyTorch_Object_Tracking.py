#------------------------------------------------------------------------------
# Settings

# Load classes
from models import *
from utils import *
from sort import *

# Other libraraies
import os, sys, time, datetime, random
import copy as cp

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import cv2
from IPython.display import clear_output

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


#------------------------------------------------------------------------------
# Filepaths and globals 

config_path='config/yolov3.cfg'
weights_path='config/yolov3.weights'
class_path='config/coco.names'
img_size=416
conf_thres=0.8
nms_thres=0.4

# Set input file
videopath = 'data/11-sample.mp4'

#------------------------------------------------------------------------------
# Load model and weights

model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
# model.cuda()
model.eval()
classes = utils.load_classes(class_path)
# Tensor = torch.cuda.FloatTensor
Tensor = torch.FloatTensor

#------------------------------------------------------------------------------
# Detection function

def detect_image(img, img_size = img_size):
    # Scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # Convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    
    # Run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    
    # Returns a Nx7 tensor, where N is the number of dectections in the imgage.
    # Each detection contains 7 elements:
    #   - 1-4 first elements are pixel coordinates
    #   - 5-6 elements accuracy (??)
    #   - 7 class
    return detections[0]


#------------------------------------------------------------------------------
# Video settings

cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

# Initialize Sort object and video capture
mot_tracker = Sort() 
vid = cv2.VideoCapture(videopath)

# Get initial frame for testing
ret, frame0 = vid.read()
frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)

# Detections on initial frame
pilimg0 = Image.fromarray(frame0)
detections0 = detect_image(pilimg0)


#------------------------------------------------------------------------------
# Video annotation functions

def detectionToPixel(
    img, 
    detection,
    img_size = img_size):
    """
    Parameters
    ----------
    img : an array image 
    detection : a (6,) array, where 4 first elements are detection coordinates, 
                5th element is track id and 6th is class id
    img_size : ??
    """
    
    # Variables to convert from detections coordiantes
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
    
    # Detection values
    x1, y1, x2, y2, obj_id, cls_pred = detection
    
    box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
    box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
    y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
    x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
    
    # Return array with top left corner of detection, box dimentions, tracking id and class
    return [x1, y1, box_h, box_w, obj_id, cls_pred]

def drawBox(img, 
            detection,
            colors = colors):
    """
    Parameters
    ----------
    img : an array image 
    detection : 
    colors :
    """
    x1, y1, box_h, box_w, obj_id, cls_pred = detection
    
    # Set colors based on detection class and class albels
    color = colors[int(obj_id) % len(colors)]
    color = [i * 255 for i in color]
    cls = classes[int(cls_pred)]
    
    # Bounding box
    cv2.rectangle(img, (x1, y1), (x1+box_w, y1+box_h), color, 2)
    # Class label rectangle
    cv2.rectangle(img, (x1, y1-20), (x1+len(cls)*15+10, y1), color, -1)
    # Class label text
    cv2.putText(img, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, .4, (255,255,255), 1)

def drawPoint(img, detection):
    x1, y1, box_h, box_w, obj_id, cls_pred = detection
    cv2.circle(img, (x1,y1), 1, (255,0,255), 2)


#------------------------------------------------------------------------------
# Process video frame by frame

tracked_objects = mot_tracker.update(detections0.cpu())

pix_detecttion_0 = detectionToPixel(frame0, tracked_objects[0])

drawBox(frame0, pix_detecttion_0)
drawPoint(frame0, pix_detecttion_0)

#while(True):
for ii in range(10):   
    # Video frame
    ret, frame_i = vid.read()
    frame_i = cv2.cvtColor(frame_i, cv2.COLOR_BGR2RGB)
    
    # Detections
    pilimg_i = Image.fromarray(frame_i)
    detections_i = detect_image(pilimg_i)
    
    pix_detecttion_i = detectionToPixel(frame_i, tracked_objects[-1])
    
    if detections_i is not None:
        tracked_objects_i = mot_tracker.update(detections_i.cpu())
        
        for obj_j in tracked_objects_i:
            pix_detecttion_j = detectionToPixel(frame_i, obj_j)
            drawBox(frame_i, pix_detecttion_j)
            # drawPoint(frame, pix_detecttion_j)
    
    # Show video
    cv2.imshow("Video", frame_i)
    # Break out by pressing 'q' when window is selected
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Make sure there are no open graphics devices
cv2.destroyAllWindows()


