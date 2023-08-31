from ultralytics import YOLO

# Path of the yolov8n.pt file
model_path = "weights/yolov8n.pt"

# path of the video file
vid_path = "videos/video_2.mp4"

# Load Pre-trained ML Model
model = YOLO(model_path)

results = model.track(vid_path,
                      conf=0.3,
                      iou=0.5,
                      persist=True,
                      show=True
                      )