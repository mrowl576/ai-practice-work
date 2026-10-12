import pandas as pd
from ultralytics import YOLO

model = YOLO("yolov8x.pt")


user_choice = input("Выберите трекер: " \
"1)botsort, " \
"2)ocsort, " \
"3)deepocsort, " \
"4)fasttrack, " \
"5)bytetrack;   ")

# Проверяем, что ввод корректен
if user_choice == "1":
    track_name = "tr/botsort.yaml"
elif user_choice == "2":
    track_name = "tr/ocsort.yaml"
elif user_choice == "3":
    track_name = "tr/deepocsort.yaml"
elif user_choice == "4":
    track_name = "tr/fasttrack.yaml"
else:
    track_name = "tr/bytetrack.yaml"  # значение по умолчанию

rows = []

for frame_idx, result in enumerate(
    model.track(
        source="input.mp4",
        tracker=track_name,
        conf=0.25,
        persist=True,
        stream=True,
        verbose=False
    ),
    start=1
):
    if result.boxes is None or result.boxes.id is None:
        continue

    boxes = result.boxes.xywh.cpu().numpy()
    ids = result.boxes.id.int().cpu().numpy()

    for box, track_id in zip(boxes, ids):
        x, y, w, h = box

        rows.append({
            "frame": frame_idx,
            "id": int(track_id),
            "x": float(x - w / 2),
            "y": float(y - h / 2),
            "w": float(w),
            "h": float(h)
        })

pd.DataFrame(
    rows,
    columns=["frame", "id", "x", "y", "w", "h"]
).to_csv("predictions.csv", index=False)
print("готово")