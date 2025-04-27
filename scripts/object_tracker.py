#!/usr/bin/env python3

import rospy
import numpy as np
import cv2 # Needed for visualization
from cv_bridge import CvBridge # Needed for visualization
from sensor_msgs.msg import Image # Needed for visualization

# --- DeepSORT Imports ---
# Ensure the DeepSORT library path is correct and accessible
import sys
# ** ADJUST THIS PATH if deep_sort is cloned elsewhere **
DEEPSORT_PATH = '/home/shehab/ros_ws/src/perception_stack/deep_sort'
try:
    sys.path.append(DEEPSORT_PATH)
    # Import from the top-level 'application_util' directory
    from application_util import preprocessing
    # Import from the inner 'deep_sort' directory
    from deep_sort import nn_matching
    from deep_sort.detection import Detection as DeepSortDetection # Rename to avoid conflict with our msg
    from deep_sort.tracker import Tracker
    rospy.loginfo(f"Successfully imported DeepSORT modules from: {DEEPSORT_PATH}")
except ImportError as e:
    rospy.logfatal(f"Failed to import DeepSORT modules. Check DEEPSORT_PATH: {DEEPSORT_PATH}. Error: {e}")
    rospy.signal_shutdown("DeepSORT import failed")
    # Exit if import fails
    sys.exit(1) # Or raise an exception that prevents node init

# --- ROS Custom Message Imports ---
# Import your custom messages from the perception_stack package
try:
    # Detections message from YOLO
    from perception_stack.msg import Detections
    # Your custom Detection message (might be reused in TrackedObjects)
    from perception_stack.msg import Detection as PerceptionDetection # Rename to avoid conflict
    # Output message containing tracked info
    from perception_stack.msg import TrackedObjects
except ImportError as e:
    rospy.logfatal(f"Failed to import custom perception_stack messages. Did you run catkin_make and source devel/setup.bash? Error: {e}")
    rospy.signal_shutdown("Custom message import failed")
    sys.exit(1)


class ObjectTracker:
    def __init__(self):
        rospy.init_node('object_tracker', anonymous=True)

        # --- DeepSORT Parameters ---
        # Get parameters from ROS param server or use defaults
        max_cosine_distance = rospy.get_param('~max_cosine_distance', 0.4)
        nn_budget = rospy.get_param('~nn_budget', None) # None for unlimited budget
        self.nms_max_overlap = rospy.get_param('~nms_max_overlap', 0.85) # High overlap threshold, assuming detector NMS is primary
        max_age = rospy.get_param('~max_age', 30) # Frames to keep unmatched track
        n_init = rospy.get_param('~n_init', 3)   # Frames to confirm a track

        # Path to the feature extraction model
        # ** ADJUST PATH if mars-small128.pb is located differently **
        model_filename = f'{DEEPSORT_PATH}/deep_sort/model_data/mars-small128.pb'
        rospy.loginfo(f"Attempting to load DeepSORT feature model from: {model_filename}")

        # Initialize DeepSORT components
        try:
            # Cosine distance metric for appearance features
            self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
            # The main tracker object
            self.tracker = Tracker(self.metric, max_age=max_age, n_init=n_init)
            rospy.loginfo("DeepSORT Tracker initialized successfully.")
            rospy.loginfo(f"Params: Max Cosine Dist={max_cosine_distance}, NN Budget={nn_budget}, NMS Overlap={self.nms_max_overlap}, Max Age={max_age}, N Init={n_init}")
        except Exception as e:
            rospy.logfatal(f"Failed to initialize DeepSORT tracker components: {e}")
            rospy.signal_shutdown("DeepSORT initialization failed")
            return # Stop initialization

        # --- Class Name Association Workaround ---
        # Dictionary to store the *last known* class name for a track ID
        # This is not perfect but the best we can do without modifying the library
        # or if tracker doesn't store the initially detected class reliably.
        self.track_id_to_class = {}

        # --- ROS Communication ---
        # Subscriber to YOLO detections
        self.detection_sub = rospy.Subscriber('/yolo_detections', Detections, self.detections_callback, queue_size=5)

        # Publisher for tracked objects message
        self.tracked_objects_pub = rospy.Publisher('/tracked_objects', TrackedObjects, queue_size=10)

        # --- Optional: For Visualization ---
        self.bridge = CvBridge()
        # Subscribe to the original camera stream to draw on
        self.image_sub = rospy.Subscriber('/camera_stream', Image, self.image_callback, queue_size=1, buff_size=2**24)
        # Publisher for the visualization image
        self.tracking_vis_pub = rospy.Publisher('/tracking_visualization', Image, queue_size=10)
        self.current_frame = None # Store the latest frame
        self.last_published_tracks = None # Store last tracks to draw

        rospy.loginfo("Object Tracker Node Initialized and Waiting for Detections.")

    def image_callback(self, msg):
        """ Stores the latest image frame for visualization """
        try:
            self.current_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            # We call visualize_tracking() after processing detections to ensure drawing uses latest tracks
        except Exception as e:
            rospy.logerr(f"Image callback error in tracker: {e}")

    def detections_callback(self, detections_msg):
        """ Processes detections from YOLO, updates tracker, and publishes results """
        if not hasattr(self, 'tracker') or self.current_frame is None:
             rospy.logwarn_throttle(5, "Tracker not initialized or frame not received yet. Skipping detection.")
             return

        header = detections_msg.header
        perception_detections = detections_msg.detections # List of PerceptionDetection messages

        if not perception_detections:
            # No detections received, predict tracker state and publish empty tracks
            rospy.logdebug("No detections received. Predicting tracker state.")
            self.tracker.predict()
            # Publish empty message or just don't publish? Publishing empty is often better.
            empty_tracks_msg = TrackedObjects(header=header)
             # Ensure lists exist even when empty if the message definition requires them
            empty_tracks_msg.ids = []
            empty_tracks_msg.classes = []
            if hasattr(empty_tracks_msg, 'tracked_detections'):
                empty_tracks_msg.tracked_detections = []
            self.tracked_objects_pub.publish(empty_tracks_msg)
            self.last_published_tracks = empty_tracks_msg # Store for visualization
            self.visualize_tracking() # Update visualization
            return

        # --- 1. Convert perception_stack/Detections to DeepSORT format ---
        bboxes_tlwh = []    # List of [x, y, w, h] for DeepSORT
        confidences = []    # List of confidences
        original_class_names = [] # Store class names corresponding to input detections by index

        for det in perception_detections:
            # Convert center_x, center_y, width, height to top-left_x, top-left_y, width, height
            tl_x = det.x - det.w / 2.0
            tl_y = det.y - det.h / 2.0
            bboxes_tlwh.append([tl_x, tl_y, det.w, det.h])
            confidences.append(det.confidence)
            original_class_names.append(det.class_name) # Keep class name associated with index

        bboxes_tlwh_np = np.array(bboxes_tlwh)
        confidences_np = np.array(confidences)

        # --- 2. Optional: Non-Max Suppression ---
        # ... (keep commented out unless needed) ...

        # --- 3. Feature Extraction (Using simplified placeholder) ---
        features = self._get_features(bboxes_tlwh_np, self.current_frame)

        # --- 4. Create DeepSORT Detection objects ---
        deep_sort_detections = []
        if len(features) == len(bboxes_tlwh_np): # Check should always pass now
             for i in range(len(bboxes_tlwh_np)):
                 bbox = bboxes_tlwh_np[i]
                 conf = confidences_np[i]
                 feat = features[i] # Will be an empty list []
                 # Create DeepSORT's specific Detection object (tlwh, confidence, feature)
                 deep_sort_detections.append(DeepSortDetection(bbox, conf, feat))
        else:
             # This case should not happen with the corrected _get_features
             rospy.logerr("CRITICAL: Mismatch between features and bboxes AFTER correction. This should not happen.")

        # --- 5. Update the Tracker ---
        self.tracker.predict() # Predict next state based on motion model
        # Only update if we have valid deep_sort detections
        if deep_sort_detections:
             # Update without the class_names keyword argument
             self.tracker.update(deep_sort_detections) # <<< CORRECTED CALL

             # --- Workaround: Update our class name dictionary ---
             # This attempts to associate the class name from the *original* detection
             # with the track ID *if* the track was just updated (matched).
             # It relies on the indices potentially matching or a simple heuristic.
             matches, _, unmatched_detections = self.tracker._match(deep_sort_detections) # Use internal match for info (may change with library versions)

             for track_idx, detection_idx in matches:
                 track_id = self.tracker.tracks[track_idx].track_id
                 # Check if detection_idx is valid for original_class_names
                 if detection_idx < len(original_class_names):
                      detected_class = original_class_names[detection_idx] # Class name from original detection
                      self.track_id_to_class[track_id] = detected_class # Update dictionary with latest matched class
                 else:
                     rospy.logwarn(f"Detection index {detection_idx} out of bounds for original class names list (len {len(original_class_names)}). Cannot update class for track {track_id}.")


        else:
             rospy.logdebug("No valid DeepSORT detections to update the tracker with (potentially due to empty input).")


        # --- 6. Process Tracker Output and Publish ---
        tracked_objects_msg = TrackedObjects()
        tracked_objects_msg.header = header # Use header from detections message

        # Clear lists before populating
        tracked_objects_msg.ids = []
        tracked_objects_msg.classes = []
        if hasattr(tracked_objects_msg, 'tracked_detections'): # Check if field exists
             tracked_objects_msg.tracked_detections = []

        active_tracks = [] # Keep track of tracks for visualization
        current_confirmed_ids = set()

        for track in self.tracker.tracks:
            # Filter out tentative tracks or those lost for too long
            if not track.is_confirmed() or track.time_since_update > 1:
                 # Clean up old class associations for tracks considered lost by the tracker
                 if track.track_id in self.track_id_to_class and track.time_since_update > self.tracker.max_age:
                      rospy.logdebug(f"Removing class association for expired track ID {track.track_id}")
                      # Use pop to safely remove, default to None if already removed
                      self.track_id_to_class.pop(track.track_id, None)
                 continue

            track_id = track.track_id
            current_confirmed_ids.add(track_id)
            active_tracks.append(track) # Add to list for visualization

            # Get the class name from our dictionary, default to "Unknown"
            # *** THE CORRECTION IS HERE ***
            class_name = self.track_id_to_class.get(track_id, "Unknown") # Use dict ONLY
            # *****************************

            # Get current estimated bounding box (TLWH format)
            bbox_tlwh = track.to_tlwh()

            # Populate the TrackedObjects message
            tracked_objects_msg.ids.append(track_id)
            tracked_objects_msg.classes.append(class_name) # Use the determined class name


            # --- Populate bounding box info if TrackedObjects.msg supports it ---
            if hasattr(tracked_objects_msg, 'tracked_detections'): # Check if field exists
              tracked_detection = PerceptionDetection()
              # Convert TLWH back to center x, y, w, h (or your msg format)
              tracked_detection.x = bbox_tlwh[0] + bbox_tlwh[2] / 2.0
              tracked_detection.y = bbox_tlwh[1] + bbox_tlwh[3] / 2.0
              tracked_detection.w = bbox_tlwh[2]
              tracked_detection.h = bbox_tlwh[3]
              tracked_detection.confidence = 1.0 # Confidence high for confirmed tracks
              tracked_detection.class_name = class_name # Use the determined class name
              tracked_objects_msg.tracked_detections.append(tracked_detection)
            # --------------------------------------------------------------------

        # Optional: More aggressive cleanup of class dictionary for tracks that disappear
        ids_to_remove = set(self.track_id_to_class.keys()) - current_confirmed_ids
        for track_id in ids_to_remove:
             # Check if they are truly old? Maybe check track.time_since_update in the main loop is enough.
             # Let's keep the cleanup minimal for now.
             pass

        # Publish the message with tracked object info
        self.tracked_objects_pub.publish(tracked_objects_msg)
        self.last_published_tracks = tracked_objects_msg # Store for visualization

        # Call visualization update
        self.visualize_tracking(active_tracks)


    def _get_features(self, tlwh_bboxes, frame):
        """
        Placeholder feature extractor. Returns empty features for each bbox.
        Ensures the output feature array count matches the input bbox count.
        """
        if not isinstance(tlwh_bboxes, np.ndarray):
             tlwh_bboxes = np.array(tlwh_bboxes) # Ensure it's a numpy array

        if tlwh_bboxes.ndim == 1: # Handle case of single bounding box
             if tlwh_bboxes.shape[0] == 0: # Empty single bbox
                 return np.array([])
             else: # Single valid bbox, reshape it
                 tlwh_bboxes = tlwh_bboxes.reshape(1, -1)

        num_bboxes = tlwh_bboxes.shape[0]

        if num_bboxes == 0:
             return np.array([]) # Return empty array if no boxes

        # Create one empty list feature for EACH input bounding box
        # This guarantees the count matches len(tlwh_bboxes)
        features = [[] for _ in range(num_bboxes)] # List of empty lists

        rospy.logdebug_throttle(10, f"Feature extraction placeholder active. Providing {len(features)} empty features for {num_bboxes} boxes.")

        # Return features matching the order and count of input bboxes
        return features


    def visualize_tracking(self, active_tracks=None):
         """ Draws bounding boxes and IDs on the current frame. """
         if self.current_frame is None:
             return # Cannot visualize without a frame

         vis_frame = self.current_frame.copy()
         header = None # Will try to get from message later if needed

         # Use provided active_tracks if available (more efficient)
         tracks_to_draw = active_tracks

         # If not provided, get from last published message (less efficient)
         if tracks_to_draw is None and self.last_published_tracks:
              header = self.last_published_tracks.header
              # Rebuild track info from message if needed (less ideal)
              if hasattr(self.last_published_tracks, 'tracked_detections') and self.last_published_tracks.tracked_detections:
                   temp_tracks = []
                   # Assuming IDs and tracked_detections lists correspond by index
                   for i, track_id in enumerate(self.last_published_tracks.ids):
                        # Check index bounds
                        if i < len(self.last_published_tracks.tracked_detections):
                             det = self.last_published_tracks.tracked_detections[i]
                             # Convert back to TLWH for drawing consistency
                             bbox_tlwh = [det.x - det.w/2.0, det.y - det.h/2.0, det.w, det.h]
                             # Get class name from our dictionary for consistency
                             class_name = self.track_id_to_class.get(track_id, det.class_name) # Use dict if available
                             temp_tracks.append({'id': track_id, 'bbox': bbox_tlwh, 'class': class_name})
                        else:
                             rospy.logwarn_throttle(10, f"Index mismatch when rebuilding tracks for visualization. ID count: {len(self.last_published_tracks.ids)}, Detection count: {len(self.last_published_tracks.tracked_detections)}")
                             break # Avoid index error
                   tracks_to_draw = temp_tracks
              else: # Cannot draw boxes if only IDs/Classes are published
                   tracks_to_draw = []
                   rospy.logwarn_throttle(10,"Cannot visualize bounding boxes as they are not in TrackedObjects message.")


         if tracks_to_draw:
             for track_info in tracks_to_draw:
                 # Adapt based on whether track_info is a Track object or dict from message
                 if hasattr(track_info, 'track_id'): # Check if it's a Track object
                     bbox_tlwh = track_info.to_tlwh()
                     track_id = track_info.track_id
                     # Get class name from our dictionary as primary source
                     # *** THE CORRECTION IS HERE ***
                     class_name = self.track_id_to_class.get(track_id, "Unknown") # Use dict only
                 elif isinstance(track_info, dict): # Check if it's a dict (from rebuilt message)
                     bbox_tlwh = track_info['bbox']
                     track_id = track_info['id']
                     class_name = track_info['class'] # Already determined when rebuilding
                 else:
                     rospy.logwarn_throttle(10, f"Unexpected track info format for visualization: {type(track_info)}")
                     continue # Skip if format is unexpected

                 # Convert tlwh to xmin, ymin, xmax, ymax for drawing
                 xmin = int(bbox_tlwh[0])
                 ymin = int(bbox_tlwh[1])
                 xmax = int(bbox_tlwh[0] + bbox_tlwh[2])
                 ymax = int(bbox_tlwh[1] + bbox_tlwh[3])

                 # Draw bounding box (e.g., blue for tracked objects)
                 cv2.rectangle(vis_frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                 label = f"ID: {track_id} ({class_name})" # Already handles None/Unknown
                 cv2.putText(vis_frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2) # Blue text

         try:
              vis_msg = self.bridge.cv2_to_imgmsg(vis_frame, 'bgr8')
              # Use header from detections if possible, otherwise use current time
              if header:
                   vis_msg.header = header
              else:
                   # If no header from message, use current time (less accurate sync)
                   vis_msg.header.stamp = rospy.Time.now()
                   # vis_msg.header.frame_id = ... # Set frame_id if known
              self.tracking_vis_pub.publish(vis_msg)
         except Exception as e:
              rospy.logerr(f"CV Bridge error during tracking visualization: {e}")


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        tracker_node = ObjectTracker()
        # Check if tracker initialization was successful before spinning
        if hasattr(tracker_node, 'tracker'):
             tracker_node.run()
        else:
            # Error already logged during __init__
            pass
    except rospy.ROSInterruptException:
        rospy.loginfo("Object Tracker node shutting down.")
    except Exception as e:
        rospy.logfatal(f"Unhandled exception in Object Tracker main: {e}")