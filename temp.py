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

# Update tracker
tracked_objects = mot_tracker.update(detections.cpu())

#-------------------------------------------------------------------
# Foo
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

pix_detecttion_0 = detectionToPixel(frame, tracked_objects[-1])

drawBox(frame, pix_detecttion_0)
drawPoint(frame, pix_detecttion_0)
