import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


model = YOLO("yolo12n.pt")


tracker = DeepSort(max_age=1000, n_init=3)


conf_threshold = 0.5


def detect_persons(frame):

    results = model(frame)
    boxes = []
    confidences = []


    for box in results[0].boxes:

        xyxy = box.xyxy.cpu().numpy()[0]
        conf = float(box.conf.cpu().numpy()[0])
        cls = int(box.cls.cpu().numpy()[0])


        if cls == 0 and conf >= conf_threshold:
            x_min, y_min, x_max, y_max = map(int, xyxy)
            boxes.append([x_min, y_min, x_max, y_max])
            confidences.append(conf)

    return boxes, confidences



video_path = 'inputvideo.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error opening video file")
    exit()


frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = 'DeepSort-Out6.mp4'
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break


    boxes, scores = detect_persons(frame)

    detections = []
    for box, score in zip(boxes, scores):
        xmin, ymin, xmax, ymax = box
        bbox = [xmin, ymin, xmax, ymax]
        detections.append([bbox, score, 0])

    tracks = tracker.update_tracks(detections, frame=frame)


    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        left, top, right, bottom = map(int, ltrb)


        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)


    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved as: {output_path}")
