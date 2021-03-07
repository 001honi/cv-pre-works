import tensorflow as tf
import cv2, time
import numpy as np

# Useful Functions
#=======================================================================================
def detect_skin(FRAME):
    roi = FRAME[ROI_H[0]:ROI_H[1],ROI_W[0]:ROI_W[1]]
    FRAME = cv2.blur(FRAME, (25,25))
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(roi_hsv, SKIN_LO, SKIN_HI)
    skinMask = cv2.dilate(skinMask, None, iterations=1)
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(roi, roi, mask = skinMask)
    FRAME[ROI_H[0]:ROI_H[1],ROI_W[0]:ROI_W[1]] = skin
    return FRAME, skin

def prepare_input(img, bgr=0):
    if not bgr:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (IMG_W, IMG_H))/255
        img = np.reshape(img, (IMG_H, IMG_W, 1))
    else:
        img = cv2.resize(img, (IMG_W, IMG_H))/255        
    return np.array([img])

def get_prediction_probs(logits, softmax=True):
    if softmax:
        softmax = tf.nn.softmax(logits).numpy()
    else:
        softmax = logits
    probabilities = []
    for prob in softmax[0]:
        p = np.array2string(prob, formatter={'float_kind':lambda x: "%.4f" % x})
        probabilities.append(float(p))
    return probabilities

def print_interface(probs, prediction):
    INTERFACE = np.zeros((480,340,3))
    figures = ("0-zero", "1-one", "2-two", "3-three", "4-four", \
        "5-five", "6-up", "7-down", "8-right", "9-left")
    line, i = 0, 0
    for prob in probs:
        if prob < 0.2:
            color = RED 
        elif prob < 0.6:
            color = ORANGE
        else:
            color = GREEN

        cv2.putText(INTERFACE, str(prob), (30,150+line), FONT, 0.6, color, 1, cv2.LINE_AA)
        cv2.rectangle(INTERFACE, (120,132+line), (int(121+prob*100), 148+line), color, -1)
        cv2.putText(INTERFACE, figures[i], (230,150+line), FONT, 0.6, color, 1, cv2.LINE_AA)
        line += 30
        i += 1

    cv2.putText(INTERFACE, "Figure Probabilities", (20,85), FONT, 1, GREEN, 2, cv2.LINE_AA)
    cv2.imshow("Predictions", INTERFACE)

def print_learning():
    BLACK = np.zeros((150,340,3))
    if LEARNING:
        cv2.putText(BLACK, "Learning Mode ON", (20,50), FONT, 1, GREEN, 2, cv2.LINE_AA)
        if cache < BATCH_SIZE+1:
            cv2.putText(BLACK, f"{cache}/{BATCH_SIZE} Caching..", (20,100), FONT, 1, ORANGE, 2, cv2.LINE_AA)
        else:
            cv2.putText(BLACK, f"OK!", (20,100), FONT, 1, GREEN, 2, cv2.LINE_AA)
            cv2.putText(BLACK, f"Reset Learning", (80,100), FONT, 1, ORANGE, 2, cv2.LINE_AA)
    else:
        cv2.putText(BLACK, "Learning Mode OFF", (20,50), FONT, 1, RED, 2, cv2.LINE_AA)
    cv2.imshow("Real-Time Learning", BLACK)



# Constants 
#=======================================================================================

# Import Model ID
#--------------------------------------------------------------------
# model_ID = 'hand-recog-cnn-model-2'
model_ID = 'hand-recog-mobile-net-v2-tf-1'
COLOR = 1   # 0: Gray , 1: BGR
CHANNEL = 3 if COLOR else 1

# Real-Time Learning
#--------------------------------------------------------------------
BATCH_SIZE    = 16
LEARNING_past = 0
cache = 0

# Image Size
#--------------------------------------------------------------------
IMG_H = 224
IMG_W = 224

# OpenCV
#--------------------------------------------------------------------
CAM = cv2.VideoCapture(0)

# Region of Interest
ROI_H = [115,365]
ROI_W = [195,445]

# Skin Detection Color Thresholds in HSV
SKIN_LO = np.array([0,48,80], dtype=np.uint8)
SKIN_HI = np.array([20,255,255], dtype=np.uint8)

# Font & Colors Constants
FONT = cv2.FONT_HERSHEY_SIMPLEX
RED     = (0, 0, 255)
GREEN   = (0, 255, 0)
BLUE    = (255, 0, 0)
ORANGE  = (0, 69, 255)
MAGENTA = (240, 0, 159)

# Create Trackbar 
nothing = lambda nothing: None
cv2.namedWindow("Real-Time Learning")
cv2.createTrackbar('Learning','Real-Time Learning',0,1,nothing)
cv2.createTrackbar('Label','Real-Time Learning',0,9,nothing)
cv2.createTrackbar('Save','Real-Time Learning',0,1,nothing)



# Load Base Model & Freeze Weights
#=======================================================================================

model = tf.keras.models.load_model(f'./models/{model_ID}')
for layer in model.layers[:-1]:
    layer.trainable = False
# model.layers[-1].trainable = False
model.summary()



# Main Loop
#=======================================================================================
while True:
    if cv2.waitKey(1) == 27:
        CAM.release()
        cv2.destroyAllWindows()
        break    
    #--------------------------------------------------------------------
    LEARNING = cv2.getTrackbarPos("Learning","Real-Time Learning")
    SAVE     = cv2.getTrackbarPos("Save","Real-Time Learning")

    _, FRAME = CAM.read()        
    FRAME, IMG = detect_skin(FRAME)
    INPUT = prepare_input(IMG, bgr=COLOR)

    # Getting Prediction via CNN Model
    #--------------------------------------------------------------------
    logits = model.predict(INPUT)
    prediction = np.argmax(logits)
    probs = get_prediction_probs(logits, softmax=True)

    # Show GUI
    #--------------------------------------------------------------------
    cv2.imshow("Hand Recognition", FRAME)
    print_interface(probs, prediction)
    print_learning()

    # Real-Time Learning
    #--------------------------------------------------------------------
    if LEARNING:
        if not LEARNING_past:
            imgs   = np.array([])
            labels = np.array([])
            cache = 0

        if cache < BATCH_SIZE:
            LABEL = cv2.getTrackbarPos("Label","Real-Time Learning")
            imgs   = np.append(imgs, INPUT[0])
            labels = np.append(labels, LABEL)
            cache += 1

        elif cache == BATCH_SIZE:
            imgs = np.reshape(imgs, (BATCH_SIZE, IMG_H, IMG_W, CHANNEL))
            model.train_on_batch(imgs, labels)
            cache += 1
    if SAVE:
        model.save(f'./models/{model_ID}-real-time-trained')

    LEARNING_past = LEARNING


