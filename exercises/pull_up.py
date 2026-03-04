# exercises/pull_up.py
import cv2
import time
from utils import angle_calculations

_STATE = {"smoothed": None, "in_down": False, "rep_count": 0, "last_rep_ts": 0.0, "last_angle": None, "last_ts": None, "tick": 0}

DOWN_THR = 60
UP_THR = 160
COOLDOWN_MS = 420
ALPHA = 0.28
MIN_VEL = 6.0

def process_frame(frame, pose_landmarks, mp_pose):
    global _STATE
    h, w = frame.shape[:2]
    lm = pose_landmarks.landmark

    try:
        l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        l_el = lm[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        l_wr = lm[mp_pose.PoseLandmark.LEFT_WRIST.value]
    except Exception:
        return frame, {"status": "landmarks_missing"}

    sh = (int(l_sh.x * w), int(l_sh.y * h))
    el = (int(l_el.x * w), int(l_el.y * h))
    wr = (int(l_wr.x * w), int(l_wr.y * h))

    cv2.circle(frame, sh, 5, (40, 200, 40), -1)
    cv2.circle(frame, el, 5, (40, 200, 40), -1)
    cv2.circle(frame, wr, 5, (40, 200, 40), -1)
    cv2.line(frame, sh, el, (200, 180, 0), 2)
    cv2.line(frame, el, wr, (200, 180, 0), 2)

    elbow_angle = angle_calculations.calculate_angle(sh, el, wr)
    angle = float(elbow_angle)
    aint = int(round(angle))

    if _STATE["smoothed"] is None:
        _STATE["smoothed"] = angle
    else:
        _STATE["smoothed"] = ALPHA * angle + (1 - ALPHA) * _STATE["smoothed"]

    now = time.time()
    vel = 0.0
    if _STATE["last_angle"] is not None and _STATE["last_ts"]:
        dt = now - _STATE["last_ts"]
        if dt > 0:
            vel = (angle - _STATE["last_angle"]) / dt
    _STATE["last_angle"] = angle
    _STATE["last_ts"] = now

    if not _STATE["in_down"]:
        if _STATE["smoothed"] <= DOWN_THR:
            _STATE["in_down"] = True
    else:
        cooldown_ok = (now - _STATE["last_rep_ts"]) * 1000 >= COOLDOWN_MS
        vel_ok = (MIN_VEL <= 0) or (abs(vel) >= MIN_VEL)
        if _STATE["smoothed"] >= UP_THR and cooldown_ok and vel_ok:
            _STATE["rep_count"] += 1
            _STATE["in_down"] = False
            _STATE["last_rep_ts"] = now

    if aint <= DOWN_THR:
        status = "bottom"
    elif aint >= UP_THR:
        status = "top"
    else:
        status = "mid"

    suggestions = []
    rep_quality = "good"

    if status == "bottom":
        if aint < 40:
            suggestions.append("Aim chin over bar — you're almost there.")
            rep_quality = "okay"
        if vel < -80:
            suggestions.append("Explosive pull — lead with elbows.")
        elif not suggestions:
            suggestions.append("Engage lats — think elbows to hips.")
    elif status == "top":
        if aint > 170:
            suggestions.append("Full hang — engage shoulders before pulling.")
        elif not suggestions:
            suggestions.append("Control the lowering — 2-3 seconds down.")
    elif status == "mid":
        if vel > 60:
            suggestions.append("Don't kip — keep movement strict.")
            rep_quality = "okay"
        elif not suggestions:
            suggestions.append("Keep core tight — no swinging.")

    if aint > 150 and _STATE["in_down"]:
        suggestions.append("Drive elbows down to initiate the pull.")

    info = {
        "elbow_angle": aint,
        "smoothed_angle": int(round(_STATE["smoothed"])),
        "status": status,
        "suggestions": suggestions[:2],
        "rep_quality": rep_quality,
        "rep_count": int(_STATE["rep_count"]),
        "last_rep_ts": _STATE["last_rep_ts"]
    }

    _STATE["tick"] = (_STATE["tick"] + 1) % 80
    if _STATE["tick"] == 0:
        print(f"[pull_up] sm={_STATE['smoothed']:.1f} a={aint} vel={vel:.1f} in_down={_STATE['in_down']} reps={_STATE['rep_count']}")

    return frame, info

def reset():
    _STATE["smoothed"] = None
    _STATE["in_down"] = False
    _STATE["rep_count"] = 0
    _STATE["last_rep_ts"] = 0.0
    _STATE["last_angle"] = None
    _STATE["last_ts"] = None
    _STATE["tick"] = 0