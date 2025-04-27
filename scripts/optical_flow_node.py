#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import message_filters # To sync image and tracks

# Import your custom messages
# TrackedObjects message (needs tracked_detections field populated)
from perception_stack.msg import TrackedObjects
# New messages for speed output
from perception_stack.msg import ObjectSpeedList, ObjectSpeed

class OpticalFlowNode:
    def __init__(self):
        rospy.init_node('optical_flow_node', anonymous=True)
        self.bridge = CvBridge()

        # --- Parameters ---
        # LK Params (if using Lucas-Kanade)
        self.lk_params = dict(winSize=(21, 21), # Larger window size can be more robust
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # Feature Detection Params (for LK features)
        self.feature_params = dict(maxCorners=20, # Max features per object
                                   qualityLevel=0.01, # Lower quality for more points
                                   minDistance=7,
                                   blockSize=7)
        # Option to use Farneback (dense) instead of Lucas-Kanade (sparse)
        self.use_farneback = rospy.get_param('~use_farneback', False)

        # --- Data Storage ---
        self.prev_gray = None
        # Dictionary: {track_id: np.array([[x1, y1]], dtype=float32)} - points from previous frame
        self.prev_tracked_points = {}
        self.last_timestamp = None

        # --- ROS Communication ---
        # Subscribers using ApproximateTimeSynchronizer
        # Queue size should be small to process recent data
        self.image_sub = message_filters.Subscriber('/camera_stream', Image)
        # TrackedObjects needs the 'tracked_detections' field with bounding boxes
        self.tracks_sub = message_filters.Subscriber('/tracked_objects', TrackedObjects)

        # Synchronizer (adjust slop based on timing differences)
        # Slop: allowed time difference in seconds
        # *** INCREASED SLOP HERE ***
        time_slop = 0.5 # Allow up to 0.5 seconds difference
        rospy.loginfo(f"Using ApproximateTimeSynchronizer with slop = {time_slop}s")
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.tracks_sub], queue_size=5, slop=time_slop
        )
        # ***************************
        self.ts.registerCallback(self.sync_callback)

        # Publisher for object speeds
        self.speed_pub = rospy.Publisher('/object_speeds', ObjectSpeedList, queue_size=10)

        # --- Optional Visualization ---
        self.flow_vis_pub = rospy.Publisher('/flow_visualization', Image, queue_size=10)
        self.current_vis_frame = None # Store frame for drawing

        rospy.loginfo("Optical Flow Node Initialized.")
        rospy.loginfo(f"Using Farneback: {self.use_farneback}")


    def sync_callback(self, image_msg, tracks_msg):
        """ Process synchronized image and track data to estimate speed """
        rospy.logdebug(f"Sync callback received. Image time: {image_msg.header.stamp.to_sec():.4f}, Tracks time: {tracks_msg.header.stamp.to_sec():.4f}")
        try:
            frame = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.current_vis_frame = frame.copy() # For visualization
        except Exception as e:
            rospy.logerr(f"CV Bridge error in sync_callback: {e}")
            return

        current_timestamp = image_msg.header.stamp
        dt = 0.0
        if self.last_timestamp is not None:
            dt = (current_timestamp - self.last_timestamp).to_sec()
            rospy.logdebug(f"Calculated dt: {dt:.4f}s")
        else:
             rospy.logdebug("First frame pair received.")


        # Avoid division by zero or nonsensical dt
        if dt <= 1e-4 and self.last_timestamp is not None: # Check only if not the first frame
             rospy.logwarn_throttle(1.0, f"Time delta too small or invalid ({dt:.4f}s). Skipping speed calculation.")
             # Update prev_gray for the next valid frame, but don't calculate speed
             self.prev_gray = gray_frame.copy()
             # Don't update last_timestamp if dt is invalid, wait for next valid pair
             return

        # --- Track Management ---
        # Get current track IDs and their bounding boxes (crucial!)
        current_bboxes_tlwh = {} # {track_id: [tl_x, tl_y, w, h]}
        current_track_ids = set()

        # ** This assumes TrackedObjects has the 'tracked_detections' field **
        if not hasattr(tracks_msg, 'tracked_detections') or not tracks_msg.tracked_detections:
             rospy.logwarn_throttle(5.0, "Received TrackedObjects message without 'tracked_detections'. Cannot estimate speed.")
             # Still update prev_gray and timestamp for next frame
             self.prev_gray = gray_frame.copy()
             self.last_timestamp = current_timestamp
             # Publish empty speed list?
             empty_speed_list = ObjectSpeedList(header=image_msg.header)
             self.speed_pub.publish(empty_speed_list)
             self.visualize_flow({}) # Publish visualization even if empty
             return

        # Extract IDs and bounding boxes
        for i, track_id in enumerate(tracks_msg.ids):
             if i < len(tracks_msg.tracked_detections):
                  det = tracks_msg.tracked_detections[i]
                  # Convert center x,y,w,h back to tlwh for ROI extraction
                  tl_x = det.x - det.w / 2.0
                  tl_y = det.y - det.h / 2.0
                  current_bboxes_tlwh[track_id] = [tl_x, tl_y, det.w, det.h]
                  current_track_ids.add(track_id)
             else:
                  rospy.logwarn(f"Index mismatch in TrackedObjects message for ID {track_id}")


        # --- Optical Flow Calculation ---
        object_speeds_list = ObjectSpeedList()
        object_speeds_list.header = image_msg.header # Use image timestamp

        new_tracked_points = {} # Store points for the *next* frame's calculation (LK only)

        # Only calculate flow if we have a previous frame
        if self.prev_gray is not None:
            rospy.logdebug("Calculating optical flow...")
            # --- Option 1: Lucas-Kanade (Sparse Flow) ---
            if not self.use_farneback:
                points_to_track_prev = [] # List of numpy arrays, one per track
                ids_for_points = [] # List of track IDs corresponding to points

                # Gather points from tracks active in the *previous* frame
                for track_id, points in self.prev_tracked_points.items():
                    # Only track points for IDs that are *still* present in the current tracks_msg
                    if track_id in current_track_ids and points is not None and len(points) > 0:
                         points_to_track_prev.append(points)
                         ids_for_points.extend([track_id] * len(points)) # Repeat ID for each point

                calculated_speeds = {} # {track_id: speed_pixels_per_sec}

                if points_to_track_prev:
                    # Flatten list of point arrays for LK input
                    p0 = np.concatenate(points_to_track_prev).astype(np.float32).reshape(-1, 1, 2)
                    rospy.logdebug(f"Tracking {len(p0)} points using LK.")

                    # Calculate optical flow
                    p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray_frame, p0, None, **self.lk_params)

                    # Filter good points (st=1 indicates tracked successfully)
                    if p1 is not None: # Check if flow calculation returned points
                        good_new = p1[st == 1]
                        good_old = p0[st == 1]
                        ids_good = np.array(ids_for_points)[st.flatten() == 1]
                        rospy.logdebug(f"Successfully tracked {len(good_new)} points.")


                        # Calculate speed per track ID based on average point displacement
                        track_displacements = {} # {track_id: [list of point distances]}
                        valid_new_points_for_next_frame = {} # {track_id: [list of valid points]}

                        for i in range(len(good_new)):
                            track_id = ids_good[i]
                            new_pt = good_new[i]
                            old_pt = good_old[i]

                            dist_pixels = np.linalg.norm(new_pt - old_pt)

                            if track_id not in track_displacements:
                                track_displacements[track_id] = []
                                valid_new_points_for_next_frame[track_id] = []

                            track_displacements[track_id].append(dist_pixels)
                            valid_new_points_for_next_frame[track_id].append(new_pt) # Store valid points

                            # --- Visualize LK Flow (Optional) ---
                            cv2.line(self.current_vis_frame, tuple(new_pt.astype(int)), tuple(old_pt.astype(int)), (0, 255, 0), 1)
                            cv2.circle(self.current_vis_frame, tuple(new_pt.astype(int)), 3, (0, 0, 255), -1)
                            # --- End Visualization ---

                        # Average speed and store points for next frame
                        for track_id, displacements in track_displacements.items():
                            if displacements:
                                avg_displacement = np.mean(displacements)
                                speed_pixels_per_sec = avg_displacement / dt # dt check already done earlier
                                calculated_speeds[track_id] = speed_pixels_per_sec
                                # Store the valid new points (reshaped) for the next iteration
                                if track_id in valid_new_points_for_next_frame:
                                    new_tracked_points[track_id] = np.array(valid_new_points_for_next_frame[track_id]).reshape(-1, 1, 2)
                                rospy.logdebug(f"Track ID {track_id}: Avg Speed = {speed_pixels_per_sec:.2f} px/s")
                            else:
                                calculated_speeds[track_id] = 0.0 # No valid points tracked for this ID

                    else: # p1 was None
                        rospy.logwarn("calcOpticalFlowPyrLK returned None for p1.")

                else: # No points to track from previous frame
                     rospy.logdebug("No previous points to track for LK.")


                # Add calculated speeds to the output message for all currently tracked IDs
                for track_id in current_track_ids:
                     speed = calculated_speeds.get(track_id, 0.0) # Default to 0 if not calculated (e.g., new track)
                     speed_msg = ObjectSpeed(id=track_id, speed=speed)
                     object_speeds_list.speeds.append(speed_msg)


            # --- Option 2: Farneback (Dense Flow) ---
            else:
                rospy.logdebug("Calculating dense flow using Farneback...")
                # Calculate dense flow between previous and current grayscale frames
                flow = cv2.calcOpticalFlowFarneback(
                    self.prev_gray, gray_frame, None,
                    pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                    poly_n=5, poly_sigma=1.2, flags=0
                )

                # --- Visualize Dense Flow (Optional) ---
                # ... (visualization code remains commented unless needed) ...

                # Estimate speed per object using average flow magnitude within its bounding box
                for track_id, bbox_tlwh in current_bboxes_tlwh.items():
                     x, y, w, h = map(int, bbox_tlwh)
                     # Ensure box coordinates are valid and within frame bounds
                     x1 = max(0, x)
                     y1 = max(0, y)
                     x2 = min(gray_frame.shape[1], x1 + w)
                     y2 = min(gray_frame.shape[0], y1 + h)

                     speed_pixels_per_sec = 0.0 # Default speed

                     # Check if the ROI is valid
                     if x2 > x1 and y2 > y1:
                         roi_flow = flow[y1:y2, x1:x2]
                         if roi_flow.size > 0:
                             # Calculate magnitude of flow vectors (sqrt(dx^2 + dy^2))
                             mag, _ = cv2.cartToPolar(roi_flow[..., 0], roi_flow[..., 1], angleInDegrees=False)
                             # Use median instead of mean to be more robust to outliers? Or mean.
                             avg_mag = np.mean(mag)
                             speed_pixels_per_sec = avg_mag / dt # dt check already done earlier
                             rospy.logdebug(f"Track ID {track_id}: Avg Flow Mag = {avg_mag:.2f}, Speed = {speed_pixels_per_sec:.2f} px/s")
                         else:
                             rospy.logdebug(f"Empty ROI flow for track {track_id}")
                     else:
                         rospy.logdebug(f"Invalid ROI for track {track_id}: [{x1},{y1},{x2-x1},{y2-y1}]")

                     # Add speed to output message
                     speed_msg = ObjectSpeed(id=track_id, speed=speed_pixels_per_sec)
                     object_speeds_list.speeds.append(speed_msg)

        else: # self.prev_gray is None (first valid frame pair)
             rospy.logdebug("prev_gray is None. Skipping flow calculation for this frame.")
             # Publish empty speeds for current tracks
             for track_id in current_track_ids:
                  object_speeds_list.speeds.append(ObjectSpeed(id=track_id, speed=0.0))


        # --- Feature Detection for Next Frame (Only for LK) ---
        if not self.use_farneback:
            rospy.logdebug("Detecting features for next LK frame...")
            # For existing tracks that had points, keep their updated points from LK (`new_tracked_points`)
            # For *new* tracks (present now but not in `new_tracked_points`), detect features.
            # Also potentially re-detect if a track lost too many points.
            final_points_for_next_frame = {}

            for track_id, bbox_tlwh in current_bboxes_tlwh.items():
                 # Check if we already have good points from LK update and if they are enough
                 if track_id in new_tracked_points and len(new_tracked_points[track_id]) >= self.feature_params['maxCorners'] // 3: # Keep if at least 1/3rd remain
                     final_points_for_next_frame[track_id] = new_tracked_points[track_id]
                     rospy.logdebug(f"Keeping {len(new_tracked_points[track_id])} tracked points for ID {track_id}")
                 else:
                     # Detect new features for this track ID (new track or lost/few points)
                     rospy.logdebug(f"Detecting new features for track ID {track_id}")
                     x, y, w, h = map(int, bbox_tlwh)
                     x1 = max(0, x)
                     y1 = max(0, y)
                     x2 = min(gray_frame.shape[1], x1 + w)
                     y2 = min(gray_frame.shape[0], y1 + h)

                     points_in_roi = None # Default
                     if x2 > x1 and y2 > y1: # Check for valid ROI
                         # Create a mask for the bounding box ROI
                         mask = np.zeros_like(gray_frame)
                         mask[y1:y2, x1:x2] = 255
                         # Detect features using goodFeaturesToTrack within the mask
                         points_in_roi = cv2.goodFeaturesToTrack(gray_frame, mask=mask, **self.feature_params)

                     if points_in_roi is not None:
                         final_points_for_next_frame[track_id] = points_in_roi
                         rospy.logdebug(f"Detected {len(points_in_roi)} new features for ID {track_id}")
                         # --- Visualize New Features (Optional) ---
                         for pt in points_in_roi:
                              cv2.circle(self.current_vis_frame, tuple(pt.ravel().astype(int)), 3, (0,255,255), -1) # Yellow
                         # --- End Visualization ---
                     else:
                          # No features detected in this box, store None or empty
                          final_points_for_next_frame[track_id] = None
                          rospy.logdebug(f"No features detected for track ID {track_id}")


            # Update the points to be used in the *next* LK calculation
            self.prev_tracked_points = final_points_for_next_frame


        # Publish the calculated speeds
        self.speed_pub.publish(object_speeds_list)
        rospy.logdebug(f"Published {len(object_speeds_list.speeds)} speed entries.")

        # Update previous frame and timestamp for the next iteration
        self.prev_gray = gray_frame.copy()
        self.last_timestamp = current_timestamp

        # Publish visualization frame
        self.visualize_flow(current_bboxes_tlwh)


    def visualize_flow(self, current_bboxes_tlwh):
        """ Publishes the visualization frame """
        if self.current_vis_frame is None:
            return

        try:
             # Draw current tracked boxes on the visualization frame for context
             for track_id, bbox_tlwh in current_bboxes_tlwh.items():
                 x, y, w, h = map(int, bbox_tlwh)
                 cv2.rectangle(self.current_vis_frame, (x, y), (x + w, y + h), (0, 0, 255), 1) # Red box for current tracks
                 cv2.putText(self.current_vis_frame, f"ID: {track_id}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

             vis_msg = self.bridge.cv2_to_imgmsg(self.current_vis_frame, 'bgr8')
             vis_msg.header.stamp = self.last_timestamp # Use the timestamp of the current frame
             # vis_msg.header.frame_id = ... # Set if known
             self.flow_vis_pub.publish(vis_msg)
        except Exception as e:
             rospy.logerr(f"CV Bridge error for flow visualization: {e}")


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = OpticalFlowNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Unhandled exception in Optical Flow Node: {e}")