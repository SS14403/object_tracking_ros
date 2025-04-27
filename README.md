# ROS Perception Stack: Detection, Tracking, and Speed Estimation

This ROS Noetic package implements a pipeline for detecting objects using YOLOv5, tracking them with DeepSORT (motion-based), and estimating their speed using optical flow.

## Pipeline Overview


*   **Nodes:** `camera_streamer`, `yolo_detector`, `object_tracker`, `optical_flow_node`, `fusion_node`
*   **Key Topics:** `/camera_stream`, `/yolo_detections`, `/tracked_objects`, `/object_speeds`, `/final_object_list`

## Requirements

*   Ubuntu 20.04
*   ROS Noetic Ninjemys
*   Python 3.8+
*   OpenCV (`python3-opencv`)
*   PyTorch & Torchvision (`pip3 install torch torchvision`)
*   Other Python packages: `numpy`, `scipy`, `pyyaml`, `pandas`, `gdown` (`pip3 install numpy scipy pyyaml pandas gdown`)
*   An external camera stream source (e.g., IP Webcam Android App)
*   (If using GUI tools in WSL) An X Server for Windows (e.g., VcXsrv) configured correctly.

## Installation & Setup

1.  **Clone this Repository:**
    ```bash
    cd ~/your_ros_ws/src
    git clone https://github.com/YourUsername/YourRepositoryName.git # Use your actual repo URL
    ```

2.  **Install External Dependencies:**
    *   **YOLOv5:**
        ```bash
        cd ~ # Or another location outside the ROS workspace src
        git clone https://github.com/ultralytics/yolov5.git
        cd yolov5
        pip3 install -r requirements.txt
        # Note: The detector node assumes this is cloned to ~/yolov5 by default.
        # Adjust path in yolo_detector.py or use ROS param _yolo_dir:=/path/to/yolov5
        ```
    *   **DeepSORT:**
        ```bash
        cd ~/your_ros_ws/src/perception_stack # Navigate INSIDE your package
        git clone https://github.com/nwojke/deep_sort.git
        cd deep_sort
        pip3 install -r requirements.txt
        # Note: The tracker node assumes this is cloned here by default.
        # Adjust DEEPSORT_PATH in object_tracker.py if cloned elsewhere.
        ```

3.  **Download Models:**
    *   **YOLOv5 Weights:** The `yolo_detector.py` node currently uses `yolov5s.pt`. This might be downloaded automatically by `torch.hub` on first run if network access is available, or you can download it manually into the `~/yolov5` directory.
    *   **DeepSORT Re-ID Model:**
        ```bash
        cd ~/your_ros_ws/src/perception_stack/deep_sort # Navigate to the cloned deep_sort dir
        mkdir -p deep_sort/model_data
        # Use gdown (recommended)
        gdown --id 18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp -O deep_sort/model_data/mars-small128.pb
        # Or download manually and copy mars-small128.pb into deep_sort/model_data/
        ```

4.  **Build the ROS Package:**
    ```bash
    cd ~/your_ros_ws
    catkin_make
    source devel/setup.bash
    ```

## Running the Pipeline

You need to run each node in a separate terminal. Make sure `roscore` is running first.

1.  **Terminal 1: `roscore`**
    ```bash
    roscore
    ```

2.  **Terminal 2: Camera Streamer** (Update URL if needed)
    ```bash
    rosrun perception_stack camera_streamer.py _camera_url:="<your_ip_webcam_url>"
    ```

3.  **Terminal 3: YOLO Detector** (Update path if needed)
    ```bash
    rosrun perception_stack yolo_detector.py _yolo_dir:=/home/your_user/yolov5
    ```

4.  **Terminal 4: Object Tracker**
    ```bash
    rosrun perception_stack object_tracker.py
    ```

5.  **Terminal 5: Optical Flow Node** (Choose LK or Farneback)
    ```bash
    # For Farneback (Dense)
    rosrun perception_stack optical_flow_node.py _use_farneback:=true
    # OR For Lucas-Kanade (Sparse)
    # rosrun perception_stack optical_flow_node.py _use_farneback:=false
    ```

6.  **Terminal 6: Fusion Node**
    ```bash
    rosrun perception_stack fusion_node.py
    ```

7.  **Terminal 7 (Optional): View Output**
    ```bash
    rostopic echo /final_object_list
    # OR view visualizations
    # rqt_image_view /yolo_visualization
    # rqt_image_view /tracking_visualization
    # rqt_image_view /flow_visualization
    ```

## Limitations & Known Issues

*   **Motion-Only Tracking:** The DeepSORT feature extraction is currently using a placeholder (empty features). Tracking relies primarily on the Kalman filter motion model and IoU, similar to SORT. Performance may degrade significantly during occlusions.
*   **Class Name Association:** Class name association in the tracker uses a workaround dictionary and may not be robust if object appearance changes significantly or if multiple objects are very close.
*   **Speed Units:** Optical flow speed is estimated in pixels/second relative to the camera frame. It does not represent real-world speed (m/s) and is affected by camera ego-motion.
*   **WSL Environment:** Developed and tested primarily in WSL 2. GUI tools require specific X Server setup. Camera access relies on network streaming.


