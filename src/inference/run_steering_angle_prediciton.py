import os
import cv2
from src.models import model
from subprocess import call

from ultralytics import YOLO
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

class SteeringAnglePredictor:
    def __init__(self, model_path):
        self.session = tf.InteractiveSession()
        self.saver = tf.train.Saver()
        self.saver.restore(self.session, model_path)
        self.smoothed_angle = 0

    def predict_angle(self, image):
        degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180.0 / 3.14159265
        return degrees

    def smooth_angle(self, predicted_angle):
        if self.smoothed_angle == 0:
            self.smoothed_angle = predicted_angle
        else:
            self.smoothed_angle += 0.2 * pow(abs(predicted_angle - self.smoothed_angle), 2.0 / 3.0) * (
                    predicted_angle - self.smoothed_angle) / abs(predicted_angle - self.smoothed_angle)
        return self.smoothed_angle

    def close(self):
        self.session.close()


class DrivingSimulator:
    def __init__(self, predictor, data_dir, steering_image_path, is_windows=False):
        self.predictor = predictor
        self.data_dir = data_dir
        self.steering_image = cv2.imread(steering_image_path, 0)
        self.is_windows = is_windows
        self.rows, self.cols = self.steering_image.shape

    def start_simulation(self):
        i = 0
        while cv2.waitKey(10) != ord('q'):
            full_image = cv2.imread(os.path.join(self.data_dir, f"{i}.jpg"))
            resized_image = cv2.resize(full_image[-150:], (200, 66)) / 255.0

            predicted_angle = self.predictor.predict_angle(resized_image)
            smoothed_angle = self.predictor.smooth_angle(predicted_angle)

            if not self.is_windows:
                call("clear")
            print(f"Predicted steering angle: {predicted_angle} degrees")

            self.display_frames(full_image, smoothed_angle)
            i += 1

        cv2.destroyAllWindows()

    def display_frames(self, full_image, smoothed_angle):
        cv2.imshow("frame", full_image)
        rotation_matrix = cv2.getRotationMatrix2D((self.cols / 2, self.rows / 2), -smoothed_angle, 1)
        rotated_steering_wheel = cv2.warpAffine(self.steering_image, rotation_matrix, (self.cols, self.rows))
        cv2.imshow("steering wheel", rotated_steering_wheel)


if __name__ == "__main__":
    model_path = "saved_models/regression_model/model.ckpt"
    data_dir = "data/driving_dataset"
    steering_image_path = "data/steering_wheel_image.jpg"

    # Determine if running on Windows
    is_windows = os.name == 'nt'

    predictor = SteeringAnglePredictor(model_path)
    simulator = DrivingSimulator(predictor, data_dir, steering_image_path, is_windows)

    try:
        simulator.start_simulation()
    finally:
        predictor.close()
