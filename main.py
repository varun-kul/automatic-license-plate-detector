# ── PIL ANTIALIAS FIX — must be FIRST ──
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

import glob
import queue
import threading
import time
import logging
import traceback
import os
from datetime import datetime

import cv2
import torch
from ultralytics import YOLO

from gate_controller import GateController
from plate_utils import PlateGateManager, MotionDetector
from snapshot import process_snapshot

torch.serialization.add_safe_globals([torch.nn.modules.container.Sequential])


# ---------------------------------------------------------------------------
# Config — edit these
# ---------------------------------------------------------------------------
source = 0              # 0=webcam | "video.mp4" | "rtsp://..." | "image.jpg"

GATE_PORT       = None        # None = auto-detect, or set e.g. 'COM5'
GATE_BAUD       = 9600
GATE_OPEN_CMD   = b'OPEN\n'
GATE_CLOSE_CMD  = b'CLOSE\n'
GATE_OPEN_SECS  = 5.0

CONFIRM_COUNT   = 1           # reads needed to confirm a plate
GATE_COOLDOWN   = 30.0        # seconds before same plate triggers again
READ_WINDOW     = 10.0        # reads must occur within this window

SNAP_COOLDOWN   = 3.0         # seconds between auto-captures
MOTION_THRESH   = 500         # pixel count to consider as motion

MAX_RESULTS_MEM = 500         # flush all_results after this many snapshots
HEALTH_LOG_SECS = 300         # log system health every N seconds
CAM_RETRY_SECS  = 5           # seconds between camera reconnect attempts
MAX_CAM_RETRIES = 10          # retries before giving up

LOG_DIR         = "./logs"


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, f"anpr_{datetime.now().strftime('%Y%m%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("ANPR")


# ---------------------------------------------------------------------------
class LiveLicensePlateDetector:
    def __init__(self):
        self.all_results = {}
        self.snap_count  = 0

        log.info("Loading model...")
        self.lp_detector = YOLO('./models/best.pt')
        self.lp_detector.conf = 0.05
        self.lp_detector.iou  = 0.3

        self.gate = PlateGateManager(
            confirm_count=CONFIRM_COUNT,
            gate_cooldown=GATE_COOLDOWN,
            read_window=READ_WINDOW
        )

        try:
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu=False)
            log.info("EasyOCR initialized")
        except ImportError:
            log.error("Install easyocr: pip install easyocr")
            self.reader = None

        self.registered_plates = self._load_registered('registered.txt')
        self.gate_ctrl         = self._init_gate()

    # ── Setup helpers ────────────────────────────────────────────────────
    def _load_registered(self, path='registered.txt') -> set:
        try:
            with open(path) as f:
                plates = {l.strip().upper() for l in f if l.strip()}
            log.info(f"Loaded {len(plates)} registered plates")
            return plates
        except FileNotFoundError:
            sample = ['MH12DE1433', 'RJ41SH7917', 'KY70CWT']
            with open(path, 'w') as f:
                f.write('\n'.join(sample))
            log.info(f"Created sample {path}")
            return set(sample)

    def _init_gate(self) -> GateController | None:
        ports = GateController.find_ports()
        log.info(f"[gate] Available ports: {ports}")
        if not ports:
            log.warning("[gate] No COM ports — gate control disabled")
            return None
        port = GATE_PORT or ports[0]
        ctrl = GateController(port=port, baudrate=GATE_BAUD,
                              open_cmd=GATE_OPEN_CMD, close_cmd=GATE_CLOSE_CMD,
                              open_duration=GATE_OPEN_SECS)
        return ctrl if ctrl.connect() else None

    def _open_camera(self, source):
        """Open camera with retry logic."""
        for attempt in range(1, MAX_CAM_RETRIES + 1):
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)
                log.info(f"Camera opened: {source}")
                return cap
            log.warning(f"Camera open failed (attempt {attempt}/{MAX_CAM_RETRIES})")
            time.sleep(CAM_RETRY_SECS)
        log.error("Could not open camera after retries")
        return None

    def _flush_old_results(self):
        """Keep memory bounded."""
        if len(self.all_results) > MAX_RESULTS_MEM:
            keys = sorted(self.all_results.keys())
            for k in keys[:len(keys) // 2]:
                del self.all_results[k]
            log.info(f"Flushed old results, kept {len(self.all_results)}")

    # ── Single image mode ────────────────────────────────────────────────
    def run_image(self, image_path):
        frame = cv2.imread(image_path)
        if frame is None:
            log.error(f"Cannot read: {image_path}"); return
        log.info(f"Image: {image_path}  {frame.shape}")
        process_snapshot(frame, self.lp_detector, self.reader,
                         self.all_results, snapshot_id=1,
                         gate=self.gate,
                         registered_plates=self.registered_plates,
                         gate_ctrl=self.gate_ctrl)
        saved = sorted(glob.glob("./detections/**/*.jpg", recursive=True),
                       key=os.path.getmtime)
        if saved:
            cv2.imshow('Result', cv2.imread(saved[-1]))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # ── Live video mode ──────────────────────────────────────────────────
    def run_live(self, video_source=0):
        cap = self._open_camera(video_source)
        if cap is None: return

        snap_queue = queue.Queue(maxsize=1)
        ocr_busy   = threading.Event()

        def ocr_worker():
            while True:
                item = snap_queue.get()
                if item is None: break
                snap, sid = item
                ocr_busy.set()
                try:
                    process_snapshot(snap, self.lp_detector, self.reader,
                                     self.all_results, sid,
                                     gate=self.gate,
                                     registered_plates=self.registered_plates,
                                     gate_ctrl=self.gate_ctrl)
                except Exception:
                    log.error(f"Snapshot crashed:\n{traceback.format_exc()}")
                finally:
                    ocr_busy.clear()

        worker = threading.Thread(target=ocr_worker, daemon=True)
        worker.start()

        motion_det    = MotionDetector(threshold=MOTION_THRESH)
        auto_capture  = True      # default ON for unattended operation
        snap_id       = 0
        last_snap_t   = 0
        last_health_t = time.time()
        start_time    = time.time()
        overlay       = []        # [(text, expire_time, is_registered)]
        fps_c, fps_t  = 0, time.time()

        log.info("Live detection started")

        try:
            while True:
                ret, frame = cap.read()

                # ── Camera reconnect ──────────────────────────────────
                if not ret:
                    log.warning("Frame read failed — reconnecting...")
                    cap.release()
                    cap = self._open_camera(video_source)
                    if cap is None:
                        log.error("Camera reconnect failed — exiting")
                        break
                    continue

                now = time.time()

                # ── Pull new detections into overlay ──────────────────
                for sid_r, fr in list(self.all_results.items()):
                    if sid_r <= snap_id - 5: continue
                    for d in fr.values():
                        txt = d['license_plate']['text']
                        reg = d['license_plate']['registered']
                        if not any(t == txt for t,_,_ in overlay):
                            overlay.append((txt, now + 8.0, reg))
                overlay = [(t,e,r) for t,e,r in overlay if e > now]

                # ── Auto-capture on motion ────────────────────────────
                has_motion = motion_det.detect(frame)
                if (auto_capture and has_motion
                        and not ocr_busy.is_set()
                        and now - last_snap_t > SNAP_COOLDOWN):
                    snap_id += 1
                    self.snap_count += 1
                    try:
                        snap_queue.put_nowait((frame.copy(), snap_id))
                        last_snap_t = now
                    except queue.Full:
                        pass

                # ── Memory flush ──────────────────────────────────────
                self._flush_old_results()

                # ── Health log every N seconds ────────────────────────
                if now - last_health_t > HEALTH_LOG_SECS:
                    uptime    = int(now - start_time)
                    hrs, rem  = divmod(uptime, 3600)
                    mins      = rem // 60
                    plates    = {d['license_plate']['text']
                                 for fr in self.all_results.values()
                                 for d in fr.values()}
                    gate_stat = 'ON' if self.gate_ctrl and self.gate_ctrl.connected else 'OFF'
                    log.info(f"HEALTH | uptime={hrs}h{mins}m "
                             f"snaps={self.snap_count} "
                             f"unique_plates={len(plates)} "
                             f"gate={gate_stat}")
                    last_health_t = now

                # ── Draw UI ───────────────────────────────────────────
                display = frame.copy()

                for i,(txt,_,reg) in enumerate(overlay[-4:]):
                    color = (0,255,0) if reg else (0,100,255)
                    cv2.putText(display, txt,
                                (display.shape[1]-320, 40+i*40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

                fps_c += 1
                fps      = fps_c / (now - fps_t + 1e-6)
                uptime_s = int(now - start_time)
                hrs, rem = divmod(uptime_s, 3600)
                mins     = rem // 60
                mode     = "AUTO" if auto_capture else "MANUAL"
                busy     = " | OCR..." if ocr_busy.is_set() else ""
                mot      = " | MOTION" if has_motion else ""
                gate_s   = 'ON' if self.gate_ctrl and self.gate_ctrl.connected else 'OFF'

                cv2.putText(display,
                            f"FPS:{fps:.0f} [{mode}]{busy}{mot} | GATE:{gate_s} | UP:{hrs}h{mins}m",
                            (10,35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)
                cv2.putText(display, "SPACE=capture  A=auto  Q=quit",
                            (10, display.shape[0]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
                cv2.circle(display, (20,70), 8,
                           (0,0,255) if has_motion else (0,255,0), -1)

                cv2.imshow('ANPR Live', display)
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    break
                elif k == ord(' ') and not ocr_busy.is_set():
                    snap_id += 1
                    self.snap_count += 1
                    try:
                        snap_queue.put_nowait((frame.copy(), snap_id))
                        last_snap_t = now
                        log.info(f"Manual capture #{snap_id}")
                    except queue.Full:
                        log.debug("OCR busy — manual capture dropped")
                elif k == ord('a'):
                    auto_capture = not auto_capture
                    log.info(f"Auto-capture: {'ON' if auto_capture else 'OFF'}")

        except Exception:
            log.error(f"Main loop crashed:\n{traceback.format_exc()}")
        finally:
            log.info("Shutting down...")
            snap_queue.put(None)
            worker.join(timeout=5)
            if self.gate_ctrl:
                self.gate_ctrl.disconnect()
            cap.release()
            cv2.destroyAllWindows()
            plates = {d['license_plate']['text']
                      for fr in self.all_results.values() for d in fr.values()}
            if plates:
                log.info(f"Session complete. Plates detected: {plates}")


# ---------------------------------------------------------------------------
def main():
    detector     = LiveLicensePlateDetector()
    video_source = source   # 0=webcam | "video.mp4" | "rtsp://..." | "image.jpg"

    log.info("=== ANPR System Starting ===")
    if isinstance(video_source, str) and \
       video_source.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tiff','.webp')):
        detector.run_image(video_source)
    else:
        detector.run_live(video_source)


if __name__ == "__main__":
    main()