import os
import csv
import time

import cv2

from plate_utils import find_plate_candidates, merge_stacked_boxes, ocr_crop, PlateGateManager
from gate_controller import GateController


def process_snapshot(frame, lp_detector, reader, results_store,
                     snapshot_id, gate: PlateGateManager,
                     registered_plates: set,
                     gate_ctrl: GateController = None):
    """
    Run YOLO + OCR on a single frame.
    Saves annotated image and updates CSV for every non-duplicate read.
    Fires gate controller on confirmed registered plates.
    """
    plate_boxes = []

    # ── YOLO detection ───────────────────────────────────────────────
    lp_out = lp_detector(frame)[0]
    for lp in lp_out.boxes.data.tolist():
        x1,y1,x2,y2,sc,_ = lp
        if sc >= 0.05:
            plate_boxes.append((x1,y1,x2,y2,sc,'model'))

    # ── Contour fallback if YOLO found nothing ───────────────────────
    if not plate_boxes:
        for x1,y1,x2,y2 in find_plate_candidates(frame):
            plate_boxes.append((x1,y1,x2,y2,0.0,'contour'))

    plate_boxes = merge_stacked_boxes(plate_boxes)

    found = {}
    for idx,(x1,y1,x2,y2,sc,src) in enumerate(plate_boxes):
        pad  = 10
        crop = frame[max(0,int(y1)-pad):min(frame.shape[0],int(y2)+pad),
                     max(0,int(x1)-pad):min(frame.shape[1],int(x2)+pad)]
        if crop.shape[0]<8 or crop.shape[1]<20:
            continue

        text, score, method = ocr_crop(reader, crop)
        if not text:
            continue

        decision      = gate.register_read(text)
        is_registered = text in registered_plates
        status        = 'REGISTERED' if is_registered else 'UNKNOWN'
        print(f"  ✓ '{text}'  score={score:.3f}  {status}  gate={decision}")

        if decision == 'duplicate':
            continue

        # ── Gate action ───────────────────────────────────────────────
        if decision == 'trigger':
            if is_registered:
                print(f"  [gate] Opening gate for '{text}'")
                if gate_ctrl:
                    gate_ctrl.open_gate(plate=text)
                else:
                    print("  [gate] Controller not connected")
            else:
                print(f"  [gate] Unknown plate '{text}' — gate NOT opened")

        # ── Save annotated image ──────────────────────────────────────
        annotated  = frame.copy()
        box_color  = (0,200,0) if is_registered else (0,0,255)
        cv2.rectangle(annotated,(int(x1),int(y1)),(int(x2),int(y2)),box_color,3)
        label = f"{text} ({'OK' if is_registered else 'UNKNOWN'}) [{decision.upper()}]"
        (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.8,2)
        cv2.rectangle(annotated,(int(x1),int(y1)-th-12),
                      (int(x1)+tw,int(y1)),box_color,-1)
        cv2.putText(annotated,label,(int(x1),int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

        ts        = int(time.time())
        plate_dir = f"./detections/{text}"
        os.makedirs(plate_dir, exist_ok=True)
        img_path  = f"{plate_dir}/{ts}_{decision}.jpg"
        cv2.imwrite(img_path, annotated)
        print(f"  Image → {img_path}")

        car_id = idx+1
        found[car_id] = {
            'car': {'bbox': [0,0,frame.shape[1],frame.shape[0]]},
            'license_plate': {
                'bbox':             [x1,y1,x2,y2],
                'text':             text,
                'bbox_score':       sc,
                'text_score':       score,
                'detection_method': method,
                'registered':       is_registered,
                'gate_decision':    decision,
                'image_path':       img_path,
            }
        }

    # ── Append to CSV ─────────────────────────────────────────────────
    if found:
        results_store[snapshot_id] = found
        csv_path   = "./results_live.csv"
        file_exists = os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp','snapshot_id','plate','registered',
                                 'gate_decision','bbox_score','text_score',
                                 'detection_method','image_path'])
            for d in found.values():
                lp = d['license_plate']
                writer.writerow([
                    int(time.time()), snapshot_id, lp['text'],
                    lp['registered'],  lp['gate_decision'],
                    round(lp['bbox_score'],3), round(lp['text_score'],3),
                    lp['detection_method'],    lp['image_path']
                ])
        print(f"  CSV updated → {csv_path}")