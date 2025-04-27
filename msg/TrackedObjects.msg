Header header
perception_stack/Detection[] tracked_detections # List of tracked detections w/ boxes
int32[] ids # Redundant if ID is added to Detection.msg, but can keep for simplicity
string[] classes # Also potentially redundant