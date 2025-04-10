import tensorflow.compat.v1 as tf
import model
import cv2
import os
import numpy as np
import colorsys
from ultralytics import YOLO
from subprocess import call
from typing import List, Tuple

tf.disable_v2_behavior()

# Check if on Windows OS
windows = os.name == 'nt'

# Set up TensorFlow session
sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "models2/saved_models/model.ckpt")

# Load steering wheel image
img = cv2.imread('steering_wheel_image.jpg', 0)
rows, cols = img.shape

# YOLO models for lane segmentation and object detection
lane_model = YOLO('models2/saved_models/best_yolo11_lane_segmentation.pt')
object_model = YOLO('models2/saved_models/yolo11s-seg.pt')

smoothed_angle = 0

def _generate_colors(num_classes: int) -> List[Tuple[int, int, int]]:
    colors = []
    for i in range(num_classes):
        hue = i / num_classes
        sat = 0.9
        val = 0.9
        rgb = colorsys.hsv_to_rgb(hue, sat, val)
        color = tuple(int(x * 255) for x in rgb)
        colors.append(color)
    return colors

def process_segmentation(img: np.ndarray, lane_model, object_model, alpha: float = 0.5) -> np.ndarray:
    """
    Process the image with lane and object detection models, then overlay results.
    """
    overlay = img.copy()
    
    # Lane detection
    lane_results = lane_model.predict(img, conf=0.5)
    for result in lane_results:
        if result.masks is None:
            continue
        for mask in result.masks.xy:
            points = np.int32([mask])
            cv2.fillPoly(overlay, points, (144, 238, 144))  # Light green for lanes
    
    # Object detection
    object_results = object_model.predict(img, conf=0.5)
    colors = _generate_colors(len(object_model.names))
    for result in object_results:
        if result.masks is None:
            continue
        for mask, box in zip(result.masks.xy, result.boxes):
            class_id = int(box.cls[0])
            color = colors[class_id]
            points = np.int32([mask])
            cv2.fillPoly(overlay, points, color)
    
    # Blend original image with overlay
    final_img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return final_img

i = 0
while cv2.waitKey(10) != ord('q'):
    # Load the current frame
    full_image = cv2.imread(f"driving_dataset/{i}.jpg")
    if full_image is None:
        print(f"Image driving_dataset/{i}.jpg not found, ending visualization.")
        break

    # Steering angle prediction
    image = cv2.resize(full_image[-150:], (200, 66)) / 255.0
    degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180.0 / 3.14159265
    
    # Clear terminal output if not on Windows
    if not windows:
        call("clear")
    print(f"Predicted steering angle: {degrees:.2f} degrees")

    # Smooth steering wheel animation
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))

    # Process segmentation for lane and object detection
    segmented_image = process_segmentation(full_image, lane_model, object_model)

    # Display the original frame, segmented frame, and steering wheel
    cv2.imshow("Original Frame", full_image)
    cv2.imshow("Segmented Frame", segmented_image)
    cv2.imshow("Steering Wheel", dst)

    i += 1

cv2.destroyAllWindows()
