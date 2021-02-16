from collections import namedtuple
import numpy as np
import cv2

#=====================================================================================
# This function copied from PyImageSearch.com
def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

# Reads the txt file and converts the data to list format
# label,x1,y1,x2,y2
def read_txt(fname):
    box_list = list()
    f = open(fname, "r")
    f_list = f.readlines()
    for line in f_list:
        label,_,_,x1,y1,x2,y2 = line.split(",")
        y2 = y2.split("\n")[0]
        x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
        box_list.append([label,x1,y1,x2,y2])
    return box_list

# It is required to sort the detections of both ground truth and predictions
# x[3]: x1 coordinate
def sort_x1(sub_li): 
    sub_li.sort(key = lambda x: x[3]) 
    return sub_li 

#=====================================================================================

gt_list = sort_x1(read_txt("gt_beyoglu.txt"))
pr_list = sort_x1(read_txt("pr_beyoglu.txt"))

# Namedtuples are used to simplify the code by help of object-like architecture
Detection = namedtuple("Detection", ["label", "gt", "pred"])

detections = list()
for i in range(len(gt_list)):
    label,x1,y1,x2,y2 = gt_list[i]
    gt = (x1,y1,x2,y2)
    label,x1,y1,x2,y2 = pr_list[i]
    pr = (x1,y1,x2,y2)
    detections.append(Detection(label,gt,pr))

image = cv2.imread("beyoglu.jpg")

for detection in detections:
    # pick random color for each object
    color = tuple(np.random.random(size=3) * 256)
    # compute the intersection over union
    iou = bb_intersection_over_union(detection.gt, detection.pred)
    # overlay is required for transparancy
    overlay = image.copy()    
	# draw the ground-truth bounding box along with the predicted bounding box
    cv2.rectangle(overlay, tuple(detection.gt[:2]), tuple(detection.gt[2:]), color, -1) 
    cv2.rectangle(image, tuple(detection.pred[:2]), tuple(detection.pred[2:]), color, 2)
    # show ground-truth-box as transparant
    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
	# write IoU score
    cv2.putText(image, "{}".format(detection.label), (detection.pred[0], detection.pred[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2) 
    cv2.putText(image, "IoU: {:.4f}".format(iou), (detection.gt[0], detection.gt[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    # print console
    print("Label: {:10s} IoU Score: {:.3f}".format(detection.label, iou))

cv2.imwrite("beyoglu_iou.jpg", image)
cv2.imshow("Image", image)
cv2.waitKey(0)