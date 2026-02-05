# exercises/jumping_jacks.py
import cv2
import time
from utils import angle_calculations
import math

_STATE = {"smoothed": None, "in_down": False, "rep_count":0, "last_rep_ts":0.0, "last_val": None, "last_ts": None, "tick":0}

# pseudo-angle thresholds
DOWN_THR = 80   # arms down
UP_THR = 140    # arms overhead
COOLDOWN_MS = 300
ALPHA = 0.30
MIN_VEL = 8.0

def _dist(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def process_frame(frame, pose_landmarks, mp_pose):
    global _STATE
    h,w = frame.shape[:2]
    lm = pose_landmarks.landmark

    try:
        l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_wr = lm[mp_pose.PoseLandmark.LEFT_WRIST.value]
        r_wr = lm[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    except Exception:
        return frame, {"status": "landmarks_missing"}

    lsh = (int(l_sh.x*w), int(l_sh.y*h))
    rsh = (int(r_sh.x*w), int(r_sh.y*h))
    lwr = (int(l_wr.x*w), int(l_wr.y*h))
    rwr = (int(r_wr.x*w), int(r_wr.y*h))

    cv2.circle(frame, lsh, 3, (40,200,40), -1)
    cv2.circle(frame, rsh, 3, (40,200,40), -1)
    cv2.circle(frame, lwr, 3, (40,200,40), -1)
    cv2.circle(frame, rwr, 3, (40,200,40), -1)

    shoulder_dist = _dist(lsh, rsh)
    avg_wrist_y = (lwr[1] + rwr[1]) / 2.0
    shoulder_y = (lsh[1] + rsh[1]) / 2.0

    pseudo = max(0, min(180, int(round(180 * (1 - (shoulder_y - avg_wrist_y) / (shoulder_dist + 1e-6))))))

    angle = float(pseudo)
    aint = int(round(angle))

    if _STATE["smoothed"] is None:
        _STATE["smoothed"] = angle
    else:
        _STATE["smoothed"] = ALPHA*angle + (1-ALPHA)*_STATE["smoothed"]

    now = time.time()
    vel = 0.0
    if _STATE["last_val"] is not None and _STATE["last_ts"]:
        dt = now - _STATE["last_ts"]
        if dt > 0:
            vel = (angle - _STATE["last_val"])/dt
    _STATE["last_val"] = angle
    _STATE["last_ts"] = now

    if not _STATE["in_down"]:
        if _STATE["smoothed"] <= DOWN_THR:
            _STATE["in_down"] = True
    else:
        cooldown_ok = (now - _STATE["last_rep_ts"])*1000 >= COOLDOWN_MS
        vel_ok = (MIN_VEL <= 0) or (abs(vel) >= MIN_VEL)
        if _STATE["smoothed"] >= UP_THR and cooldown_ok and vel_ok:
            _STATE["rep_count"] += 1
            _STATE["in_down"] = False
            _STATE["last_rep_ts"] = now

    status = "arms_down" if aint <= DOWN_THR else ("arms_up" if aint >= UP_THR else "mid")
    suggestions = []
    info = {
        "angle": aint,
        "smoothed_angle": int(round(_STATE["smoothed"])),
        "status": status,
        "suggestions": suggestions,
        "rep_count": int(_STATE["rep_count"]),
        "last_rep_ts": _STATE["last_rep_ts"]
    }

    _STATE["tick"] = (_STATE["tick"] + 1) % 60
    if _STATE["tick"] == 0:
        print(f"[jj] sm={_STATE['smoothed']:.1f} a={aint} vel={vel:.1f} in_down={_STATE['in_down']} reps={_STATE['rep_count']}")

    return frame, info
