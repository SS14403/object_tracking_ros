#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
import cv2
import numpy as np
# Import your custom messages
from perception_stack.msg import Detection, Detections # <<< ADD THIS
import std_msgs.msg # <<< ADD THIS for Header

class YoloDetector:
    def __init__(self):
        rospy.init_node('yolo_detector', anonymous=True)
        self.bridge = CvBridge()

        # --- Model Loading ---
        # Get parameters from ROS param server or use defaults
        # Example command-line override: _yolo_dir:="/path/to/my/yolov5"
        yolo_base_dir = rospy.get_param('~yolo_dir', '/home/shehab/yolov5') # Default path
        weights_path = f"{yolo_base_dir}/yolov5s.pt" # Using standard small weights
        confidence_threshold = rospy.get_param('~confidence_threshold', 0.4)
        iou_threshold = rospy.get_param('~iou_threshold', 0.45)

        rospy.loginfo(f"Loading YOLOv5 model from: {yolo_base_dir}")
        rospy.loginfo(f"Using weights: {weights_path}")
        rospy.loginfo(f"Confidence Threshold: {confidence_threshold}")
        rospy.loginfo(f"IOU Threshold: {iou_threshold}")

        try:
            # Load model locally from the cloned repository
            self.model = torch.hub.load(yolo_base_dir,
                                   'custom', # Need 'custom' for local weights
                                   path=weights_path,
                                   source='local', # Specify source as local
                                   force_reload=False) # Avoid re-downloading if cached
            self.model.conf = confidence_threshold # Set confidence threshold
            self.model.iou = iou_threshold      # Set IoU threshold
            # self.model.classes = [0] # Optional: Filter for specific classes (e.g., 0 for 'person' in COCO)

            rospy.loginfo("YOLOv5 model loaded successfully.")

        except Exception as e:
             rospy.logerr(f"Failed to load YOLOv5 model: {e}")
             rospy.logwarn(f"Check if yolov5 repo exists at '{yolo_base_dir}' and weights '{weights_path}' are present.")
             rospy.signal_shutdown("Failed to load YOLOv5 model") # Exit if model fails
             return # Stop initialization

        # --- ROS Communication ---
        # Subscriber for incoming images
        self.image_sub = rospy.Subscriber('/camera_stream', Image, self.image_callback, queue_size=1, buff_size=2**24)

        # Publisher for the structured Detections message
        self.detections_pub = rospy.Publisher('/yolo_detections', Detections, queue_size=10) # <<< CHANGED Publisher Type

        # Publisher for the visualization image (optional)
        self.vis_pub = rospy.Publisher('/yolo_visualization', Image, queue_size=10) # <<< ADDED Visualization Publisher

        rospy.loginfo("YOLOv5 Detector Node Initialized and Waiting for Images.")


    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            rospy.logerr(f"CV Bridge error converting image: {e}")
            return

        try:
            # Perform inference
            results = self.model(frame)

            # Parse results using pandas format for easy access
            df = results.pandas().xyxy[0] # BBoxes: xmin, ymin, xmax, ymax, confidence, class, name

            # Create the Detections message to publish
            detections_msg = Detections()
            detections_msg.header = msg.header # Use the same timestamp and frame_id as input image

            # Create a clean frame for visualization (optional)
            vis_frame = frame.copy()

            # Iterate through detected objects
            for i in range(len(df)):
                # Extract data for each detection
                xmin = float(df.loc[i, 'xmin'])
                ymin = float(df.loc[i, 'ymin'])
                xmax = float(df.loc[i, 'xmax'])
                ymax = float(df.loc[i, 'ymax'])
                confidence = float(df.loc[i, 'confidence'])
                # class_id = int(df.loc[i, 'class']) # Integer class ID if needed
                class_name = df.loc[i, 'name']

                # Create a single Detection message
                detection = Detection()

                # --- Populate Detection message fields ---
                # ** IMPORTANT: Adjust this based on your exact Detection.msg definition **
                # Assuming your Detection.msg uses center x, y, width, height:
                detection.w = xmax - xmin
                detection.h = ymax - ymin
                detection.x = xmin + detection.w / 2.0 # Center X
                detection.y = ymin + detection.h / 2.0 # Center Y
                # ----------------------------------------
                detection.confidence = confidence
                detection.class_name = class_name

                # Add the single detection to the list in the Detections message
                detections_msg.detections.append(detection)

                # --- Draw on visualization frame (Optional) ---
                cv2.rectangle(vis_frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2) # Green box
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(vis_frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # Green text

            # Publish the structured Detections message
            self.detections_pub.publish(detections_msg)

            # Publish the visualization image (Optional)
            try:
                 vis_msg = self.bridge.cv2_to_imgmsg(vis_frame, 'bgr8')
                 vis_msg.header = msg.header # Keep timestamp consistent
                 self.vis_pub.publish(vis_msg)
            except Exception as e:
                 rospy.logerr(f"CV Bridge error converting/publishing visualization: {e}")

        except Exception as e:
            rospy.logerr(f"Error during YOLO inference or processing: {e}")


    def run(self):
         # Keep the node running
         rospy.spin()

if __name__ == '__main__':
    try:
        # Initialize the node object
        detector = YoloDetector()
        # Start the ROS loop only if initialization was successful (model loaded)
        if hasattr(detector, 'model'):
            detector.run()
        else:
            # Initialization failed (error already logged)
            pass
    except rospy.ROSInterruptException:
        # Normal shutdown (Ctrl+C)
        rospy.loginfo("YOLO Detector node shutting down.")
    except Exception as e:
        # Catch any unexpected errors during node setup or shutdown
        rospy.logfatal(f"Unhandled exception in YOLO Detector main: {e}")