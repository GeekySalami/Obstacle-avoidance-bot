import cv2
import numpy as np
import time

class ObjectTracker:
    def __init__(self, prototxt_path, model_path, known_dimensions, focal_length=700):
        # Load pre-trained model and configuration
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        self.classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", 
                        "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", 
                        "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
        self.known_dimensions = known_dimensions
        self.DEFAULT_DIMENSION = 0.5  # Default dimension for unknown objects
        self.focal_length = focal_length
        self.previous_distances = {}  # For calculating speed
        self.previous_times = {}  # For calculating speed

    def estimate_distance(self, known_dimension, pixel_size):
        return (self.focal_length * known_dimension) / pixel_size

    def estimate_speed(self, object_id, current_distance):
        current_time = time.time()
        speed = None

        if object_id in self.previous_distances:
            previous_distance = self.previous_distances[object_id]
            previous_time = self.previous_times[object_id]
            time_diff = current_time - previous_time
            distance_diff = previous_distance - current_distance
            speed = distance_diff / time_diff  # Speed in meters per second (m/s)

        # Update previous values
        self.previous_distances[object_id] = current_distance
        self.previous_times[object_id] = current_time

        return speed

    def detect_and_track(self, frame):
        # Prepare the frame for the neural network
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()

        results = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:  # Filter weak detections
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")

                # Calculate the pixel width and height of the object
                pixel_width = endX - startX
                pixel_height = endY - startY
                object_class = self.classes[idx]
                known_dimension = self.known_dimensions.get(object_class, self.DEFAULT_DIMENSION)

                # Calculate distance based on pixel size
                pixel_size = max(pixel_width, pixel_height)  # Take the larger dimension
                distance = self.estimate_distance(known_dimension, pixel_size)

                # Estimate object speed (using object class as ID for simplicity)
                speed = self.estimate_speed(object_class, distance)

                # Append results for the current object
                results.append({
                    "class": object_class,
                    "confidence": confidence,
                    "bounding_box": (startX, startY, endX, endY),
                    "distance": distance,
                    "speed": speed
                })

        return results
