#!/usr/bin/env python3

import rospy
import message_filters
from perception_stack.msg import TrackedObjects, ObjectSpeedList, ObjectSpeed
# Import the new final output messages
from perception_stack.msg import FinalObjectList, FinalObject

class FusionNode:
    def __init__(self):
        rospy.init_node('fusion_node', anonymous=True)

        # --- Subscribers using ApproximateTimeSynchronizer ---
        # Subscribe to the tracker output (needs IDs and Classes at minimum)
        tracks_sub = message_filters.Subscriber('/tracked_objects', TrackedObjects)
        # Subscribe to the speed estimation output (needs IDs and Speeds)
        speeds_sub = message_filters.Subscriber('/object_speeds', ObjectSpeedList)

        # Synchronize the topics
        # Adjust queue size and slop as needed based on pipeline latency
        # Slop needs to be large enough to account for delays between tracker and speed estimator
        time_slop = rospy.get_param("~time_slop", 0.5) # Allow 0.5s difference by default
        rospy.loginfo(f"Using ApproximateTimeSynchronizer with slop = {time_slop}s")
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [tracks_sub, speeds_sub], queue_size=10, slop=time_slop
        )
        self.ts.registerCallback(self.sync_callback)

        # Publisher for the final fused object list
        self.final_pub = rospy.Publisher('/final_object_list', FinalObjectList, queue_size=10)

        rospy.loginfo("Fusion Node Initialized and Waiting for Data.")

    def sync_callback(self, tracks_msg, speeds_msg):
        """ Combines synchronized track and speed information. """
        rospy.logdebug(f"Fusion callback received. Tracks time: {tracks_msg.header.stamp.to_sec():.4f}, Speeds time: {speeds_msg.header.stamp.to_sec():.4f}")

        # Process the inputs to easily access data by track ID
        track_info = {} # Dictionary: {track_id: {'class': class_name, 'bbox': bbox_obj (optional)}}
        for i, track_id in enumerate(tracks_msg.ids):
            class_name = "Unknown"
            if i < len(tracks_msg.classes):
                class_name = tracks_msg.classes[i] if tracks_msg.classes[i] else "Unknown"

            bbox_data = None
            if hasattr(tracks_msg, 'tracked_detections') and i < len(tracks_msg.tracked_detections):
                bbox_data = tracks_msg.tracked_detections[i] # Store the whole Detection msg

            track_info[track_id] = {'class': class_name, 'bbox': bbox_data}

        speed_info = {speed_obj.id: speed_obj.speed for speed_obj in speeds_msg.speeds}

        # --- Fusion Logic ---
        # Find track IDs that are present in both messages
        common_ids = set(track_info.keys()) & set(speed_info.keys())
        rospy.logdebug(f"Common Track IDs found: {common_ids}")


        final_object_list_msg = FinalObjectList()
        # Use the later timestamp from the synchronized messages for the output header
        final_object_list_msg.header = tracks_msg.header if tracks_msg.header.stamp >= speeds_msg.header.stamp else speeds_msg.header

        for track_id in common_ids:
            try:
                # Create a FinalObject message
                final_object = FinalObject()
                final_object.id = track_id
                final_object.class_label = track_info[track_id]['class']
                final_object.speed = speed_info[track_id]

                # --- Optional: Add BBox center ---
                # if track_info[track_id]['bbox'] is not None:
                #     # Assuming geometry_msgs/Point position is added to FinalObject.msg
                #     from geometry_msgs.msg import Point
                #     final_object.position = Point(x=track_info[track_id]['bbox'].x,
                #                                   y=track_info[track_id]['bbox'].y,
                #                                   z=0.0) # z=0 if only 2D
                # ------------------------------------

                # Append the fused object to the list
                final_object_list_msg.objects.append(final_object)

            except KeyError as e:
                 rospy.logwarn(f"KeyError during fusion for ID {track_id}: {e}. This might happen with rapidly changing tracks.")
            except Exception as e:
                 rospy.logerr(f"Error creating FinalObject for ID {track_id}: {e}")

        # Publish the final fused list
        self.final_pub.publish(final_object_list_msg)
        rospy.logdebug(f"Published {len(final_object_list_msg.objects)} fused objects.")


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = FusionNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Fusion node interrupted.")
    except Exception as e:
        rospy.logfatal(f"Unhandled exception in Fusion Node main: {e}")
