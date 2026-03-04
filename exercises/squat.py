# exercises/squat.py
import cv2
import time

from streamlit import status
from utils import angle_calculations

_STATE = {"smoothed": None, "in_down": False, "rep_count": 0, "last_rep_ts": 0.0, "last_angle": None, "last_ts": None, "tick": 0}

# tuning
DOWN_THR = 95    # angle <= this => down (knee angle small = deep)
UP_THR = 150     # angle >= this => up
COOLDOWN_MS = 500
ALPHA = 0.28
MIN_VEL = 4.0    # deg/sec

def process_frame(frame, pose_landmarks, mp_pose):
    global _STATE
    h, w = frame.shape[:2]
    lm = pose_landmarks.landmark

    try:
        r_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
        r_knee = lm[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        r_ankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    except Exception:
        return frame, {"status": "landmarks_missing"}

    hip = (int(r_hip.x*w), int(r_hip.y*h))
    knee = (int(r_knee.x*w), int(r_knee.y*h))
    ankle = (int(r_ankle.x*w), int(r_ankle.y*h))

    cv2.circle(frame, hip, 5, (40,200,40), -1)
    cv2.circle(frame, knee, 5, (40,200,40), -1)
    cv2.circle(frame, ankle, 5, (40,200,40), -1)
    cv2.line(frame, hip, knee, (200,180,0), 2)
    cv2.line(frame, knee, ankle, (200,180,0), 2)

    knee_angle = angle_calculations.calculate_angle(hip, knee, ankle)
    angle = float(knee_angle)
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

    # state machine
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
    rep_quality = "good"

# Range of motion
    if _STATE["in_down"] and aint > 110:
        suggestions.append("Go deeper — aim for thighs parallel to floor.")
        rep_quality = "needs work"
    elif aint < 60:
        suggestions.append("Control depth — protect your knees.")
        rep_quality = "okay"

# Phase-specific cues
    if status == "down":
        if vel < -80:  # dropping too fast (negative = descending)
            suggestions.append("Slow down the descent — 2-3 seconds down.")
            rep_quality = "okay"
        elif not suggestions:
            suggestions.append("Drive through your heels to stand.")
    elif status == "up":
        if not suggestions:
            suggestions.append("Stand tall — squeeze glutes at the top.")
    elif status == "mid":
        if vel > 0 and aint < 130:
            suggestions.append("Keep chest up as you rise.")

# Knee tracking (rough check — if knee angle is very asymmetric this fires less reliably)
    if aint <= DOWN_THR and aint > 70:
        suggestions.append("Push knees out over toes.")

    info = {
        "knee_angle": aint,
    "smoothed_angle": int(round(_STATE["smoothed"])),
    "status": status,
    "suggestions": suggestions[:2],  # max 2 at a time
    "rep_quality": rep_quality,
    "rep_count": int(_STATE["rep_count"]),
    "last_rep_ts": _STATE["last_rep_ts"]
}

    _STATE["tick"] = (_STATE["tick"] + 1) % 80
    if _STATE["tick"] == 0:
        print(f"[squat] sm={_STATE['smoothed']:.1f} a={aint} vel={vel:.1f} in_down={_STATE['in_down']} reps={_STATE['rep_count']}")

    return frame, info
def reset():
    """Reset rep counter and state machine (called by /reset_reps endpoint)."""
    _STATE["smoothed"] = None
    _STATE["in_down"] = False
    _STATE["rep_count"] = 0
    _STATE["last_rep_ts"] = 0.0
    _STATE["last_angle"] = None  # use "last_val" for jumping_jacks.py
    _STATE["last_ts"] = None
