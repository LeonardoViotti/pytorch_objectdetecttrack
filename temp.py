import copy as cp


# Functions
def img_show(img, max_t = 10000):
    # Kill window if Q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        v2.destroyAllWindows()
    
    # Show image
    cv2.imshow('Preview',img)
    
    # Otherwise kill after max_t, default 10s
    cv2.waitKey(max_t)
    cv2.destroyAllWindows()

# Initialize Sort object and video capture
mot_tracker = Sort() 
vid = cv2.VideoCapture(videopath)


# Get initial frame for testing
ret, frame = vid.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Copy initial frame for comparison
frame0 = cp.deepcopy(frame)

# Detections
pilimg = Image.fromarray(frame)
detections = detect_image(pilimg)

# img_show(frame)
# img_show(img)

# Image processing to get padded values. Because detections are done in this scale (???)

img = np.array(pilimg) # To bem concencido que isso aqui vai pro saco. Por que diabos outra instancia dessa imagem se ela e igualzinha a frame???

pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
unpad_h = img_size - pad_y
unpad_w = img_size - pad_x

# Update tracker
tracked_objects = mot_tracker.update(detections.cpu())

# Get number of different classes in detections. WHY AGAIN?        
unique_labels = detections[:, -1].cpu().unique()
n_cls_preds = len(unique_labels)

# for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
x1, y1, x2, y2, obj_id, cls_pred = tracked_objects[-1]

box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])

color = colors[int(obj_id) % len(colors)]
color = [i * 255 for i in color]
cls = classes[int(cls_pred)]

# Bounding box
cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 2)

# Class label rectangle
# cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+60, y1), color, -1)
cv2.rectangle(frame, (x1, y1-20), (x1+len(cls)*15+10, y1), color, -1)

# Class label text
cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, .4, (255,255,255), 1)

img_show(frame)

detections[0]
tracked_objects[-1]


#-------------------------------------------------------------------
# cv2.circle(frame, (detections[0][0], detections[0][1]), 1, (255,0,255), 2)
cv2.circle(frame, (detections[0][0], detections[0][1]), 1, (255,0,255), 2)

img_show(frame)


#-------------------------------------------------------------------
# Set bbox video anotation function
def drawBox(img, 
            detection,
            img_size = img_size,
            classes = classes):
    """
    Parameters
    ----------
    img : a numpy.ndarray image 
    detection : a 1x7 torch.Tensor, where 4 first elements are detection coordinates
    img_size : ??
    classes : ordered list of coco classes
    """
    
    # Variables to convert from detections coordiantes
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
    
    # Detection values
    x1, y1, x2, y2, obj_id, cls_pred = tracked_objects[-1]
    
    box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
    box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
    y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
    x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
    
    # Set colors based on detection class and class albels
    color = colors[int(obj_id) % len(colors)]
    color = [i * 255 for i in color]
    cls = classes[int(cls_pred)]
    
    #------------------------------------------------------
    # Anotations
    
    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 2)
    
    # Class label rectangle
    cv2.rectangle(frame, (x1, y1-20), (x1+len(cls)*15+10, y1), color, -1)
    
    # Class label text
    cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, .4, (255,255,255), 1)


drawBox(frame, tracked_objects[-1])
img_show(frame)

