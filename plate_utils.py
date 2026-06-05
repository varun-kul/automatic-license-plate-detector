import re
import time
from collections import deque

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Validation & scoring
# ---------------------------------------------------------------------------
INDIAN_STATE_CODES = {
    'AN','AP','AR','AS','BR','CH','CG','DN','DD','DL','GA',
    'GJ','HR','HP','JK','JH','KA','KL','LD','MP','MH','MN',
    'ML','MZ','NL','OD','PY','PB','RJ','SK','TN','TS','TR',
    'UP','UK','WB','TG'
}

def validate_indian_plate(text):
    if not text: return False
    clean = re.sub(r'[^A-Z0-9]', '', text.upper())
    if len(clean) < 6 or len(clean) > 12: return False
    if not any(clean.startswith(c) for c in INDIAN_STATE_CODES): return False
    patterns = [
        r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{1,4}$',
        r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$',
        r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[A-Z0-9]{4,5}$',
    ]
    return any(re.match(p, clean) for p in patterns)

def score_plate(text):
    if not text: return 0.0
    if validate_indian_plate(text): return 1.5
    clean = re.sub(r'[^A-Z0-9]', '', text.upper())
    if 6 <= len(clean) <= 12: return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
SKIP_TOKENS = {'IND','EU','GB','UK','USA','AUS','CAN','NZ'}

def strip_color_sidebar(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = image.shape[:2]
    ls = np.mean(hsv[:, :max(1, int(w*0.15)), 1])
    rs = np.mean(hsv[:, min(w-1, int(w*0.85)):, 1])
    x0 = int(w*0.15) if ls > 60 else 0
    x1 = int(w*0.85) if rs > 60 else w
    return image[:, x0:x1] if (x0 or x1 < w) else image

def enhance_plate(image):
    results = []
    image = strip_color_sidebar(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape)==3 else image.copy()
    h, w = gray.shape
    sf = max(120/h, 400/w)
    if sf > 1.0:
        gray = cv2.resize(gray, (int(w*sf), int(h*sf)), interpolation=cv2.INTER_CUBIC)
    results.append(("gray", gray))
    results.append(("denoised", cv2.fastNlMeansDenoising(gray, h=10)))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    results.append(("clahe", clahe.apply(gray)))
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    results.append(("otsu", otsu))
    results.append(("otsu_inv", cv2.bitwise_not(otsu)))
    results.append(("adaptive", cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)))
    results.append(("sharpened", cv2.filter2D(gray, -1,
        np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))))
    bil = cv2.bilateralFilter(gray, 11, 17, 17)
    _, bil_o = cv2.threshold(bil, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    results.append(("bilateral_otsu", bil_o))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    results.append(("morph_close", cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, k2)))
    return results


# ---------------------------------------------------------------------------
# Two-line merge
# ---------------------------------------------------------------------------
def merge_two_line(detections):
    if len(detections) < 2: return None, 0
    def cy(d): return sum(p[1] for p in d[0]) / len(d[0])
    lines = []
    for d in sorted(detections, key=cy):
        if len(d) < 3: continue
        cleaned = re.sub(r'[^A-Z0-9]', '', d[1].upper())
        if cleaned in SKIP_TOKENS or len(cleaned) < 2: continue
        lines.append((cleaned, d[2]))
    if len(lines) < 2: return None, 0
    combined = ''.join(l[0] for l in lines)
    score = sum(l[1] for l in lines) / len(lines)
    return combined, score


# ---------------------------------------------------------------------------
# Stacked box merge (two-line plates detected as two separate boxes)
# ---------------------------------------------------------------------------
def merge_stacked_boxes(plate_boxes):
    if len(plate_boxes) < 2: return plate_boxes
    merged, used = [], [False]*len(plate_boxes)
    for i in range(len(plate_boxes)):
        if used[i]: continue
        x1i,y1i,x2i,y2i,sci,srci = plate_boxes[i]
        wi,hi = x2i-x1i, y2i-y1i
        best_j, best_gap = -1, float('inf')
        for j in range(i+1, len(plate_boxes)):
            if used[j]: continue
            x1j,y1j,x2j,y2j,scj,_ = plate_boxes[j]
            wj,hj = x2j-x1j, y2j-y1j
            if min(x2i,x2j)-max(x1i,x1j) < 0: continue
            if abs(wi-wj)/float(max(wi,wj,1)) > 0.4: continue
            gap = max(y1i,y1j)-min(y2i,y2j)
            if gap > max(hi,hj)*0.5: continue
            if gap < best_gap: best_gap,best_j = gap,j
        if best_j != -1:
            x1j,y1j,x2j,y2j,scj,_ = plate_boxes[best_j]
            merged.append((min(x1i,x1j),min(y1i,y1j),max(x2i,x2j),max(y2i,y2j),
                           max(sci,scj),'merged_2line'))
            used[i] = used[best_j] = True
        else:
            merged.append(plate_boxes[i]); used[i] = True
    for i in range(len(plate_boxes)):
        if not used[i]: merged.append(plate_boxes[i])
    return merged


# ---------------------------------------------------------------------------
# Contour-based plate candidate finder
# ---------------------------------------------------------------------------
def find_plate_candidates(frame):
    fh, fw = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    candidates = []
    for bk,lo,hi in [(5,50,150),(3,30,100),(7,60,200)]:
        blurred = cv2.GaussianBlur(gray,(bk,bk),0)
        edged = cv2.Canny(blurred,lo,hi)
        dil = cv2.dilate(edged, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
        for cnt in cv2.findContours(dil,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]:
            x,y,w,h = cv2.boundingRect(cnt)
            if w<60 or h<15: continue
            asp=w/float(h); ar=(w*h)/float(fw*fh)
            if 1.8<asp<7.5 and 0.005<ar<0.30:
                candidates.append((x,y,x+w,y+h,1.0/(1+abs(asp-4))))
    _,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    for cnt in cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]:
        x,y,w,h = cv2.boundingRect(cnt)
        if w<60 or h<15: continue
        asp=w/float(h); ar=(w*h)/float(fw*fh)
        if 1.8<asp<7.5 and 0.005<ar<0.30:
            candidates.append((x,y,x+w,y+h,1.0/(1+abs(asp-4))))
    if not candidates: return []
    candidates.sort(key=lambda c:c[4], reverse=True)
    filtered = []
    for box in candidates:
        x1,y1,x2,y2,_ = box
        keep = True
        for fx1,fy1,fx2,fy2 in filtered:
            ix=max(0,min(x2,fx2)-max(x1,fx1)); iy=max(0,min(y2,fy2)-max(y1,fy1))
            if ix*iy/float((x2-x1)*(y2-y1)+(fx2-fx1)*(fy2-fy1)-ix*iy+1e-6)>0.3:
                keep=False; break
        if keep: filtered.append((x1,y1,x2,y2))
        if len(filtered)>=8: break
    return filtered


# ---------------------------------------------------------------------------
# OCR one crop
# ---------------------------------------------------------------------------
def ocr_crop(reader, crop):
    enhanced = enhance_plate(crop)
    best_text, best_score, best_method = None, 0.0, None
    for name, img in enhanced:
        try:
            dets = []
            for w_ths,h_ths in [(0.3,0.3),(0.5,0.5),(0.7,0.7),(0.9,0.9)]:
                dets = reader.readtext(img, width_ths=w_ths, height_ths=h_ths,
                                       paragraph=False, detail=1,
                                       allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                if dets: break
            if not dets: continue
            if len(dets) >= 2:
                merged, mscore = merge_two_line(dets)
                if merged:
                    mult = score_plate(merged)
                    adj = (mscore+0.5)*mult if mult > 0 else 0
                    if adj > best_score:
                        best_text,best_score,best_method = merged,adj,name+"_merged"
            for det in dets:
                if len(det) != 3: continue
                _,text,score = det
                cleaned = re.sub(r'[^A-Z0-9]','',text.upper())
                if len(cleaned) < 3: continue
                mult = score_plate(cleaned)
                if mult == 0: continue
                adj = score * mult
                if adj > best_score:
                    best_text,best_score,best_method = cleaned,adj,name
        except Exception as e:
            print(f"  OCR error [{name}]: {e}")
    return best_text, best_score, best_method


# ---------------------------------------------------------------------------
# Gate manager — dedup, confirm count, cooldown
# ---------------------------------------------------------------------------
class PlateGateManager:
    def __init__(self, confirm_count=2, gate_cooldown=30.0, read_window=10.0):
        self.confirm_count = confirm_count
        self.gate_cooldown = gate_cooldown
        self.read_window   = read_window
        self.read_times:    dict[str, deque] = {}
        self.last_triggered:dict[str, float] = {}

    def register_read(self, text) -> str:
        now  = time.time()
        text = text.upper().strip()
        last = self.last_triggered.get(text, 0)
        if now - last < self.gate_cooldown:
            remaining = int(self.gate_cooldown - (now - last))
            print(f"  [gate] '{text}' duplicate — cooldown {remaining}s")
            return 'duplicate'
        if text not in self.read_times:
            self.read_times[text] = deque()
        self.read_times[text] = deque(
            [t for t in self.read_times[text] if now-t < self.read_window])
        self.read_times[text].append(now)
        count = len(self.read_times[text])
        print(f"  [gate] '{text}' read {count}/{self.confirm_count}")
        if count >= self.confirm_count:
            self.last_triggered[text] = now
            self.read_times[text].clear()
            print(f"  [gate] ✓ '{text}' CONFIRMED")
            return 'trigger'
        return 'pending'

    def reset(self, text=None):
        if text:
            self.last_triggered.pop(text, None)
            self.read_times.pop(text, None)
        else:
            self.last_triggered.clear()
            self.read_times.clear()


# ---------------------------------------------------------------------------
# Motion detector
# ---------------------------------------------------------------------------
class MotionDetector:
    def __init__(self, threshold=500):
        self.prev_gray = None
        self.threshold = threshold

    def detect(self, frame):
        gray = cv2.GaussianBlur(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21,21), 0)
        if self.prev_gray is None:
            self.prev_gray = gray
            return False
        diff = cv2.absdiff(self.prev_gray, gray)
        self.prev_gray = gray
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        return cv2.countNonZero(thresh) > self.threshold