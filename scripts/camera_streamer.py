#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import std_msgs.msg # Needed to set header info potentially, though cv_bridge usually adds it

def stream_camera():
    rospy.init_node('camera_streamer', anonymous=True)
    # Consider adding latch=True if you want the last published image to be available
    # to late subscribers, although usually not needed for continuous streams.
    pub = rospy.Publisher('/camera_stream', Image, queue_size=10)

    # Get frame rate from parameter or use default
    frame_rate = rospy.get_param("~frame_rate", 30) # Default to 30 FPS
    rate = rospy.Rate(frame_rate)
    rospy.loginfo(f"Streaming at approximately {frame_rate} FPS")

    bridge = CvBridge()

    # Get camera stream URL from parameter or use default
    # Example override: _camera_url:="http://192.168.1.100:8080/video"
    camera_url = rospy.get_param("~camera_url", "http://192.168.1.11:8080/video")
    rospy.loginfo(f"Connecting to camera stream: {camera_url}")

    # Change this to your Windows stream (from OBS, IP Webcam, etc.)
    cap = cv2.VideoCapture(camera_url)

    # --- Optional: Add simple reconnection logic ---
    max_retries = 5
    retry_delay = 2.0 # seconds
    retry_count = 0
    # ---

    while not rospy.is_shutdown():
        if not cap.isOpened():
            rospy.logwarn(f"Video stream not open. Retrying connection... ({retry_count+1}/{max_retries})")
            cap.release() # Release previous attempt
            rospy.sleep(retry_delay)
            cap = cv2.VideoCapture(camera_url)
            retry_count += 1
            if retry_count >= max_retries:
                rospy.logerr(f"Failed to connect to video stream after {max_retries} retries. Shutting down.")
                break # Exit loop if connection fails repeatedly
            continue # Go back to start of loop to check if opened

        # Reset retry count if connection is successful
        retry_count = 0

        ret, frame = cap.read()
        if not ret:
            rospy.logwarn("Failed to grab frame, stream might have ended or stalled. Will attempt reconnect.")
            cap.release() # Release the potentially broken stream
            continue # Loop back to attempt reconnection

        if frame is None or frame.size == 0:
             rospy.logwarn("Grabbed empty frame. Skipping.")
             continue

        try:
            # Create the message using cv_bridge
            msg = bridge.cv2_to_imgmsg(frame, encoding='bgr8')

            # *** THE IMPORTANT CHANGE IS HERE ***
            # Get the current ROS time and assign it to the message header
            msg.header.stamp = rospy.Time.now()
            # Optional: Set the frame_id if you have one (e.g., 'camera_link')
            # msg.header.frame_id = "ip_webcam"
            # ***********************************

            pub.publish(msg)

        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
        except Exception as e:
            rospy.logerr(f"Error processing or publishing frame: {e}")

        try:
            rate.sleep()
        except rospy.ROSInterruptException:
            rospy.loginfo("ROS Interrupt received, shutting down streamer.")
            break # Exit loop on Ctrl+C

    rospy.loginfo("Releasing video capture device.")
    cap.release()

if __name__ == '__main__':
    try:
        stream_camera()
    except rospy.ROSInterruptException:
        rospy.loginfo("Camera streamer node interrupted.")
    except Exception as e:
        rospy.logfatal(f"Unhandled exception in camera streamer: {e}")