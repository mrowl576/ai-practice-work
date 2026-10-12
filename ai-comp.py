import numpy as np
import pandas as pd
import motmetrics as mm

gt = pd.read_csv("gt.csv")
pred = pd.read_csv("predictions.csv")

required = {"frame", "id", "x", "y", "w", "h"}
assert required.issubset(gt.columns)
assert required.issubset(pred.columns)

acc = mm.MOTAccumulator(auto_id=False)

frames = sorted(set(gt["frame"]) | set(pred["frame"]))

for frame in frames:
    g = gt[gt["frame"] == frame]
    p = pred[pred["frame"] == frame]

    gt_ids = g["id"].to_numpy(dtype=int)
    pred_ids = p["id"].to_numpy(dtype=int)

    gt_boxes = g[["x", "y", "w", "h"]].to_numpy()
    pred_boxes = p[["x", "y", "w", "h"]].to_numpy()

    # Сопоставление рамок по IoU.
    distances = mm.distances.iou_matrix(
        gt_boxes,
        pred_boxes,
        max_iou=0.5
    )

    acc.update(
        gt_ids,
        pred_ids,
        distances,
        frameid=int(frame)
    )

mh = mm.metrics.create()

summary = mh.compute(
    acc,
    metrics=[
        "idf1",
        "mota",
        "motp",
        "num_switches",
        "num_misses",
        "num_false_positives",
        "num_fragmentations",
    ],
    name="YOLO"
)

print(summary.T.to_string())

row = summary.iloc[0]

print(f"\nIDF1: {row['idf1']:.4f} ")
print(f"MOTA: {row['mota']:.4f} ")
print(f"ID-switches: {int(row['num_switches'])}")
print(f"Потери объектов (FN): {int(row['num_misses'])}")
print(f"Фрагментации траекторий: "
      f"{int(row['num_fragmentations'])}")