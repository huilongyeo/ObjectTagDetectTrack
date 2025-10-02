import cv2
from ultralytics import YOLO
from sort import Sort  
import numpy as np

video_path = 'sample.mp4'
output_path = 'output.mp4'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

person_model = YOLO('yolov8s.pt')
staff_tag_model = YOLO('runs/detect/train/weights/best.pt')
tracker = Sort()
staff_tag_memory = dict()

frame_count = 0

def get_staff_tag_conf(crop_img, conf_thres=0.6):
    if crop_img is None or crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
        return None
    results = staff_tag_model(crop_img)
    boxes = results[0].boxes
    if len(boxes) > 0:
        best_conf = float(boxes[0].conf[0])
        for box in boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
        if best_conf >= conf_thres:
            return best_conf
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detect human(get bbox)
    results = person_model(frame)[0]
    box_idx = 0
    dets = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id == 0 and conf >= 0.4:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            dets.append([x1, y1, x2, y2, conf])

    dets = np.array(dets)
    if dets.shape[0] == 0:
        dets = np.empty((0, 5))
    # id assign by SORT
    tracked_objs = tracker.update(np.array(dets))

    for track in tracked_objs:
        x1, y1, x2, y2, track_id = map(int, track[:5])
        heigh, width = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(heigh, y2)
        if x2 > x1 and y2 > y1:
            crop_img = frame[y1:y2, x1:x2]
            if crop_img.shape[0] > 0 and crop_img.shape[1] > 0:
                staff_results = get_staff_tag_conf(crop_img)
                has_tag = staff_results is not None

            if has_tag:
                staff_tag_memory[track_id] = True
        
            if staff_tag_memory.get(track_id, False):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f'Staff_Tag', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    out_writer.write(frame)
    frame_count += 1

cap.release()
out_writer.release()
print("Completed")