# exercises/crunches.py
import cv2
import time
from utils import angle_calculations

_STATE = {"smoothed": None, "in_down": False, "rep_count": 0, "last_rep_ts":0.0, "last_angle": None, "last_ts": None, "tick":0}

DOWN_THR = 45    # low hip angle => crunched
UP_THR = 120
COOLDOWN_MS = 420
ALPHA = 0.30
MIN_VEL = 3.0

def process_frame(frame, pose_landmarks, mp_pose):
    global _STATE
    h,w = frame.shape[:2]
    lm = pose_landmarks.landmark

    try:
        l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        l_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
        l_knee = lm[mp_pose.PoseLandmark.LEFT_KNEE.value]
    except Exception:
        return frame, {"status": "landmarks_missing"}

    sh = (int(l_sh.x*w), int(l_sh.y*h))
    hip = (int(l_hip.x*w), int(l_hip.y*h))
    knee = (int(l_knee.x*w), int(l_knee.y*h))

    cv2.circle(frame, sh, 4, (40,200,40), -1)
    cv2.circle(frame, hip, 4, (40,200,40), -1)
    cv2.circle(frame, knee, 4, (40,200,40), -1)
    cv2.line(frame, sh, hip, (200,180,0), 2)
    cv2.line(frame, hip, knee, (200,180,0), 2)

    hip_angle = angle_calculations.calculate_angle(sh, hip, knee)
    angle = float(hip_angle)
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
        status = "crunched"
    elif aint >= UP_THR:
        status = "extended"
    else:
        status = "mid"

    suggestions = []
    if aint < 30:
        suggestions.append("Avoid jerking neck; lift with abs.")
    if aint > 140:
        suggestions.append("Squeeze at top for better contraction.")

    info = {
        "hip_angle": aint,
        "smoothed_angle": int(round(_STATE["smoothed"])),
        "status": status,
        "suggestions": suggestions,
        "rep_count": int(_STATE["rep_count"]),
        "last_rep_ts": _STATE["last_rep_ts"]
    }

    _STATE["tick"] = (_STATE["tick"] + 1) % 80
    if _STATE["tick"] == 0:
        print(f"[crunches] sm={_STATE['smoothed']:.1f} a={aint} vel={vel:.1f} in_down={_STATE['in_down']} reps={_STATE['rep_count']}")

    return frame, info
