# exercises/dips.py
import cv2
import time
from utils import angle_calculations

_STATE = {"smoothed": None, "in_down": False, "rep_count": 0, "last_rep_ts": 0.0, "last_angle": None, "last_ts": None, "tick": 0}

DOWN_THR = 70
UP_THR = 150
COOLDOWN_MS = 480
ALPHA = 0.32
MIN_VEL = 5.0

def process_frame(frame, pose_landmarks, mp_pose):
    global _STATE
    h,w = frame.shape[:2]
    lm = pose_landmarks.landmark

    try:
        r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        r_el = lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        r_wr = lm[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    except Exception:
        return frame, {"status": "landmarks_missing"}

    sh = (int(r_sh.x*w), int(r_sh.y*h))
    el = (int(r_el.x*w), int(r_el.y*h))
    wr = (int(r_wr.x*w), int(r_wr.y*h))

    cv2.circle(frame, sh, 4, (40,200,40), -1)
    cv2.circle(frame, el, 4, (40,200,40), -1)
    cv2.circle(frame, wr, 4, (40,200,40), -1)
    cv2.line(frame, sh, el, (200,180,0), 2)
    cv2.line(frame, el, wr, (200,180,0), 2)

    elbow_angle = angle_calculations.calculate_angle(sh, el, wr)
    angle = float(elbow_angle)
    aint = int(round(angle))

    if _STATE["smoothed"] is None:
        _STATE["smoothed"] = angle
    else:
        _STATE["smoothed"] = ALPHA*angle + (1-ALPHA)*_STATE["smoothed"]

    now = time.time()
    vel = 0.0
    if _STATE["last_angle"] is not None and _STATE["last_ts"]:
        dt = now - _STATE["last_ts"]
        if dt > 0:
            vel = (angle - _STATE["last_angle"])/dt
    _STATE["last_angle"] = angle
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

    status = "unknown"
    if aint <= DOWN_THR:
        status = "down"
    elif aint >= UP_THR:
        status = "up"
    else:
        status = "mid"

    suggestions = []
    if aint < 50:
        suggestions.append("Protect shoulders; don't go too deep.")
    if aint > 165:
        suggestions.append("Control extension; avoid full lockout.")

    info = {
        "elbow_angle": aint,
        "smoothed_angle": int(round(_STATE["smoothed"])),
        "status": status,
        "suggestions": suggestions,
        "rep_count": int(_STATE["rep_count"]),
        "last_rep_ts": _STATE["last_rep_ts"]
    }

    _STATE["tick"] = (_STATE["tick"] + 1) % 100
    if _STATE["tick"] == 0:
        print(f"[dips] sm={_STATE['smoothed']:.1f} a={aint} vel={vel:.1f} in_down={_STATE['in_down']} reps={_STATE['rep_count']}")

    return frame, info
