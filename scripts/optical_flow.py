#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from perception_stack.msg import TrackedObjects  # From previous step

class OpticalFlowNode:
    def __init__(self):
        rospy.init_node('optical_flow')
        self.bridge = CvBridge()
        
        # Subscribers
        self.sub_image = rospy.Subscriber('/camera_stream', Image, self.image_callback)
        self.sub_tracks = rospy.Subscriber('/tracked_objects', TrackedObjects, self.tracks_callback)
        
        # Publishers
        self.pub_flow = rospy.Publisher('/object_flow', TrackedObjects, queue_size=10)
        
        # Optical flow setup (using Farneback method)
        self.prev_frame = None
        self.prev_points = None
        self.tracked_objects = None

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame is not None and self.prev_points is not None:
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Estimate speed for each tracked object
            for i, (x, y) in enumerate(self.prev_points):
                dx, dy = flow[int(y), int(x)]
                speed = np.sqrt(dx**2 + dy**2)
                self.tracked_objects.speeds[i] = speed  # Add 'speeds' field to TrackedObjects.msg

            self.pub_flow.publish(self.tracked_objects)
        
        self.prev_frame = gray

    def tracks_callback(self, msg):
        self.tracked_objects = msg
        self.prev_points = [(bbox.x, bbox.y) for bbox in msg.detections]  # Use centroid of bounding boxes

if __name__ == '__main__':
    OpticalFlowNode()
    rospy.spin()
