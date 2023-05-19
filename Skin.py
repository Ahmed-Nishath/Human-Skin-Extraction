import cv2
import numpy as np
import mediapipe as mp
from matplotlib import pyplot as plt

## Input Image
file_name = "Sample_3.png"
path = "Images/" + file_name
img = cv2.imread(path, 1)

## Media Pipe Initialization
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

## Resizing the image
scale_percent = 50
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dsize = (width, height)
img = cv2.resize(img, dsize)

## Background Extraction
RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
segmentation_results = selfie_segmentation.process(RGB)
mp_mask = segmentation_results.segmentation_mask

BG_COLOR = (0, 0, 0)
## Extract the human
condition = np.stack((segmentation_results.segmentation_mask,) * 3, axis=-1) > 0.1
bg_image = np.zeros(img.shape, dtype=np.uint8)
bg_image[:] = BG_COLOR
segmentation_output = np.where(condition, img, bg_image)

#_______________________________________________________________________________________________

potrait= segmentation_output
potrait_orginal = potrait.copy()

## Convert to YCR_CB Color Model
im_ycrcb = cv2.cvtColor(potrait, cv2.COLOR_BGR2YCR_CB)

## Define the Color Ranges for Skin
skin_ycrcb_mint = np.array((40, 133, 77))
skin_ycrcb_maxt = np.array((255, 173, 127))
skin_ycrcb = cv2.inRange(im_ycrcb, skin_ycrcb_mint, skin_ycrcb_maxt)

## Draw Contours for Skin
contours, _ = cv2.findContours(skin_ycrcb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for i, c in enumerate(contours):
    area = cv2.contourArea(c)
    if area > 1000:
        cv2.drawContours(potrait_orginal, contours, i, (0, 255, 0), 3)

final_result= cv2.bitwise_and(potrait,potrait, mask= skin_ycrcb)

#_______________________________________________________________________________________________

## Identify the Skin Tone
_, mask_thresh = cv2.threshold(skin_ycrcb, 127, 255, cv2.THRESH_BINARY)
positive_pixel = np.sum(mask_thresh == 255)

thresh_result= cv2.bitwise_and(potrait,potrait, mask= mask_thresh)

avg_skin_total= [0, 0, 0]
for x in range(mask_thresh.shape[0]):
    for y in range(mask_thresh.shape[1]):
        avg_skin_total[0] += final_result[x][y][0]
        avg_skin_total[1] +=  final_result[x][y][1]
        avg_skin_total[2] += final_result[x][y][2]

average_skin_tone = np.round(avg_skin_total/positive_pixel)

skin = np.zeros([300, 300, 3], dtype=np.uint8)
skin[:, :, 0] = average_skin_tone[0]
skin[:, :, 1] = average_skin_tone[1]
skin[:, :, 2] = average_skin_tone[2]

## Clasify the Race
h, s, v = cv2.split(cv2.cvtColor(skin, cv2.COLOR_BGR2HSV))

if v[1, 1] > 160:
    race = "Race: White"
elif 160 > v[1, 1] > 150:
    race = "Race: Brown"
else:
    race = "Race: Black"

cv2.putText(skin, race, (60, 140), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2)

#_______________________________________________________________________________________________

## Displaying the Output
plt.subplot(231), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image", fontsize=7), plt.axis('off')

plt.subplot(232), plt.imshow(mp_mask, 'gray')
plt.title("Segmentation Mask", fontsize=7), plt.axis('off')

plt.subplot(233), plt.imshow(cv2.cvtColor(segmentation_output, cv2.COLOR_BGR2RGB))
plt.title("MediaPipe Segmentation", fontsize=7), plt.axis('off')

plt.subplot(234), plt.imshow(cv2.cvtColor(potrait_orginal, cv2.COLOR_BGR2RGB))
plt.title("Contours for Skin", fontsize=7), plt.axis('off')

plt.subplot(235), plt.imshow(skin_ycrcb, 'gray')
plt.title("Skin Color Mask", fontsize=7), plt.axis('off')

plt.subplot(236), plt.imshow(cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB))
plt.title("Final Output", fontsize=7), plt.axis('off')
plt.show()

cv2.imshow("Skin Color", skin)

plt.imshow(cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB))
plt.title("Final Result", fontsize=7), plt.axis('off')
plt.show()
