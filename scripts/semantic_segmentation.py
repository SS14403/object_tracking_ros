#!/usr/bin/env python3

import rospy
import torch
import torchvision
import torchvision.transforms as T
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class SegmentationNode:
    def __init__(self):
        rospy.loginfo("=== Initializing SegmentationNode ===")
        
        # Initialize ROS
        rospy.init_node('semantic_segmentation', anonymous=True)
        self.bridge = CvBridge()
        
        # Debug: Check attributes step-by-step
        rospy.loginfo("1. ROS initialized")
        
        # Load model
        try:
            self.model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
            rospy.loginfo("2. Model loaded")
        except Exception as e:
            rospy.logerr(f"Model failed: {e}")
            raise
        
        # Create preprocessing
        self.preprocess = T.Compose([
            T.ToPILImage(),
            T.Resize((240, 320)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        rospy.loginfo(f"3. Preprocess exists: {hasattr(self, 'preprocess')}")
        
        # ROS pub/sub
        self.sub = rospy.Subscriber('/camera_stream', Image, self.callback)
        self.pub = rospy.Publisher('/segmented_image', Image, queue_size=10)
        rospy.loginfo("4. Ready to segment")

    def callback(self, msg):
        try:
            rospy.loginfo("Callback started")
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            input_tensor = self.preprocess(frame).unsqueeze(0)
            
            with torch.no_grad():
                output = self.model(input_tensor)['out'][0]
            
            seg = output.argmax(0).byte().cpu().numpy()
            color_seg = cv2.applyColorMap((seg * 10).astype(np.uint8), cv2.COLORMAP_JET)
            seg_msg = self.bridge.cv2_to_imgmsg(color_seg, encoding='bgr8')
            self.pub.publish(seg_msg)
        except Exception as e:
            rospy.logerr(f"Callback error: {e}")

if __name__ == '__main__':
    try:
        SegmentationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
