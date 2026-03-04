# exercises/bench_press.py
import cv2
import time
from utils import angle_calculations

_STATE = {
    "smoothed_angle": None,
    "in_down": False,
    "rep_count": 0,
    "last_rep_ts": 0.0,
    "last_angle": None,
    "last_angle_ts": None,
    "debug_tick": 0
}

DOWN_THRESHOLD = 75
UP_THRESHOLD = 150
COOLDOWN_MS = 450
ALPHA = 0.28
MIN_VELOCITY = 5.0

def _choose_arm(landmarks, mp_pose):
    try:
        l_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_sh = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_vis = getattr(l_sh, "visibility", None)
        r_vis = getattr(r_sh, "visibility", None)
        if l_vis is not None and r_vis is not None:
            return "left" if l_vis >= r_vis else "right"
        return "left" if abs(l_sh.x - 0.5) < abs(r_sh.x - 0.5) else "right"
    except Exception:
        return "right"

def process_frame(frame, pose_landmarks, mp_pose):
    global _STATE
    h, w = frame.shape[:2]
    lm = pose_landmarks.landmark

    side = _choose_arm(lm, mp_pose)

    try:
        if side == "left":
            sh_lm = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            el_lm = lm[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            wr_lm = lm[mp_pose.PoseLandmark.LEFT_WRIST.value]
        else:
            sh_lm = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            el_lm = lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            wr_lm = lm[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        shoulder = (int(sh_lm.x * w), int(sh_lm.y * h))
        elbow    = (int(el_lm.x * w), int(el_lm.y * h))
        wrist    = (int(wr_lm.x * w), int(wr_lm.y * h))
    except Exception:
        return frame, {"status": "landmarks_missing"}

    cv2.circle(frame, shoulder, 4, (40, 200, 40), -1)
    cv2.circle(frame, elbow,    4, (40, 200, 40), -1)
    cv2.circle(frame, wrist,    4, (40, 200, 40), -1)
    cv2.line(frame, shoulder, elbow, (200, 180, 0), 2)
    cv2.line(frame, elbow, wrist,    (200, 180, 0), 2)

    elbow_angle = angle_calculations.calculate_angle(shoulder, elbow, wrist)
    ea = float(elbow_angle)
    ea_int = int(round(ea))

    if _STATE["smoothed_angle"] is None:
        _STATE["smoothed_angle"] = ea
    else:
        _STATE["smoothed_angle"] = ALPHA * ea + (1 - ALPHA) * _STATE["smoothed_angle"]

    now = time.time()
    vel = 0.0
    if _STATE["last_angle"] is not None and _STATE["last_angle_ts"] is not None:
        dt = now - _STATE["last_angle_ts"]
        if dt > 0:
            vel = (ea - _STATE["last_angle"]) / dt
    _STATE["last_angle"] = ea
    _STATE["last_angle_ts"] = now

    s = _STATE

    if not s["in_down"]:
        if s["smoothed_angle"] <= DOWN_THRESHOLD:
            s["in_down"] = True
    else:
        cooldown_ok = (now - s["last_rep_ts"]) * 1000.0 >= COOLDOWN_MS
        vel_ok = True if MIN_VELOCITY <= 0 else abs(vel) >= MIN_VELOCITY
        if s["smoothed_angle"] >= UP_THRESHOLD and cooldown_ok and vel_ok:
            s["rep_count"] += 1
            s["in_down"] = False
            s["last_rep_ts"] = now

    if ea_int <= DOWN_THRESHOLD:
        status = "down"
    elif ea_int >= UP_THRESHOLD:
        status = "up"
    else:
        status = "mid"

    suggestions = []
    rep_quality = "good"

    if status == "down":
        if ea_int > 85:
            suggestions.append("Lower the bar to mid-chest for full ROM.")
            rep_quality = "needs work"
        if vel < -100:
            suggestions.append("Don't bounce — controlled descent.")
            rep_quality = "okay"
        elif not suggestions:
            suggestions.append("Brief pause at chest — no bounce.")
    elif status == "up":
        if ea_int > 170:
            suggestions.append("Soft lockout — keep tension in chest.")
        elif not suggestions:
            suggestions.append("Drive bar up and slightly back over shoulders.")
    elif status == "mid":
        if not suggestions:
            suggestions.append("Brace your core and keep feet flat.")

    if ea_int < 60:
        suggestions.append("Controlled descent — avoid bar bounce.")
        rep_quality = "needs work"

    info = {
        "elbow_angle": ea_int,
        "smoothed_angle": int(round(s["smoothed_angle"])),
        "status": status,
        "suggestions": suggestions[:2],
        "rep_quality": rep_quality,
        "rep_count": int(s["rep_count"]),
        "last_rep_ts": s["last_rep_ts"]
    }

    s["debug_tick"] = (s["debug_tick"] + 1) % 60
    if s["debug_tick"] == 0:
        print(f"[bench_press] smoothed={s['smoothed_angle']:.1f} angle={ea_int} vel={vel:.1f} in_down={s['in_down']} rep_count={s['rep_count']}")

    return frame, info

def reset():
    _STATE["smoothed_angle"] = None
    _STATE["in_down"] = False
    _STATE["rep_count"] = 0
    _STATE["last_rep_ts"] = 0.0
    _STATE["last_angle"] = None
    _STATE["last_angle_ts"] = None
    _STATE["debug_tick"] = 0