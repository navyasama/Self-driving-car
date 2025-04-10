import cv2
import os
from ultralytics import YOLO
import numpy as np
import colorsys
from typing import List, Tuple

class SegmentationVisualizer:
    def __init__(self, model_path_1: str, model_path_2: str, conf_threshold: float = 0.5):
        self.model_1 = YOLO(model_path_1)
        self.model_2 = YOLO(model_path_2)
        self.conf_threshold = conf_threshold
        self.class_names_1 = self.model_1.names
        self.class_names_2 = self.model_2.names
        self.colors_2 = self._generate_colors(len(self.class_names_2))
        self.lane_color = (144, 238, 144)

    def _generate_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
        colors = []
        for i in range(num_classes):
            hue = i / num_classes
            sat = 0.9
            val = 0.9
            rgb = colorsys.hsv_to_rgb(hue, sat, val)
            color = tuple(int(x * 255) for x in rgb)
            colors.append(color)
        return colors

    def _get_lane_color(self, class_id: int) -> Tuple[int, int, int]:
        class_name = self.class_names_1[class_id]
        if class_name.lower() == "lane1":
            return self.lane_color
        else:
            return (0, 255, 0)

    def process_image(self, img: np.ndarray, alpha: float = 0.5, show_labels: bool = True,
                      show_conf: bool = True, show_boxes: bool = True, box_thickness: int = 2) -> np.ndarray:
        overlay = img.copy()
        results_1 = self.model_1.predict(img, conf=self.conf_threshold)
        overlay = self._apply_model_results(overlay, results_1, self.class_names_1, None, alpha, show_labels=False,
                                            show_conf=False, show_boxes=False, box_thickness=box_thickness,
                                            lane_model=True)
        results_2 = self.model_2.predict(img, conf=self.conf_threshold)
        final_img = self._apply_model_results(overlay, results_2, self.class_names_2, self.colors_2, alpha, show_labels,
                                              show_conf=show_conf, show_boxes=show_boxes, box_thickness=box_thickness,
                                              lane_model=False)
        return final_img

    def _apply_model_results(self, overlay, results, class_names, colors, alpha, show_labels, show_conf, show_boxes,
                             box_thickness, lane_model=False):
        for result in results:
            if result.masks is None:
                continue

            for mask, box in zip(result.masks.xy, result.boxes):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = class_names[class_id]

                if lane_model:
                    color = self._get_lane_color(class_id)
                else:
                    color = colors[class_id]

                points = np.int32([mask])
                cv2.fillPoly(overlay, points, color)

                if show_boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, box_thickness)

                if show_labels or show_conf:
                    label_x, label_y = (x1, y1) if show_boxes else (int(min(mask[:, 0])), int(min(mask[:, 1])))
                    label = [class_name] if show_labels else []
                    if show_conf:
                        label.append(f"{confidence:.2f}")
                    label_text = " ".join(label)
                    (label_w, label_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(overlay, (label_x, label_y - 20), (label_x + label_w, label_y), color, -1)
                    cv2.putText(overlay, label_text, (label_x, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.addWeighted(overlay, alpha, overlay, 1 - alpha, 0, overlay)
        return overlay


def display_images_with_segmentation(input_folder, display_time=20):
    image_files = [img for img in os.listdir(input_folder) if img.endswith(".jpg") or img.endswith(".png")]
    
    # Modify this for presentation
    # image_files = [img for img in image_files if 30000 <= int(img.split('.')[0]) <= 35000]
    image_files.sort(key=lambda x: int(x.split('.')[0]))

    visualizer = SegmentationVisualizer(
        model_path_1="models/saved_models/best_yolo11_lane_segmentation.pt",
        model_path_2="models/saved_models/yolo11s-seg.pt",
        conf_threshold=0.5
    )

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)
        if image is not None:
            segmented_image = visualizer.process_image(image)

            cv2.imshow('Original Image Viewer', image)
            cv2.imshow('Segmented Image Viewer', segmented_image)
            print(f"Displaying {image_file}")

            if cv2.waitKey(display_time) == ord('q'):
                break
        else:
            print(f"Failed to load {image_file}")

    cv2.destroyAllWindows()


# Usage example
input_folder = 'driving_dataset'  # Replace with the path to your images directory
display_images_with_segmentation(input_folder, display_time=100)  # Images are displayed for 2000 milliseconds (2 seconds)
