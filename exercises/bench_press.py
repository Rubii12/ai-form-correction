# exercises/bench_press.py
import cv2
import time
from utils import angle_calculations

# Module-level state so counting persists across frames
_STATE = {
    "smoothed_angle": None,
    "in_down": False,
    "rep_count": 0,
    "last_rep_ts": 0.0,
    "last_angle": None,
    "last_angle_ts": None,
    "debug_tick": 0
}

# Tunable parameters (adjust if counting is too sensitive / slow)
DOWN_THRESHOLD = 75      # angle <= this considered "down" (bar near chest)
UP_THRESHOLD = 150       # angle >= this considered "up" (arms extended)
COOLDOWN_MS = 450        # minimum ms between counted reps
ALPHA = 0.28             # EMA smoothing factor for angle
MIN_VELOCITY = 5.0       # deg/sec - optional: require some movement speed (set 0 to disable)

def _choose_arm(landmarks, mp_pose):
    """Pick left or right based on visibility or center proximity."""
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
    """
    Annotate frame lightly and return (frame, info)
    info contains: elbow_angle (int), status (str), suggestions (list),
                   rep_count (int), last_rep_ts (float epoch seconds)
    """
    global _STATE
    h, w = frame.shape[:2]
    lm = pose_landmarks.landmark

    side = _choose_arm(lm, mp_pose)

    try:
        if side == "left":
            sh_lm = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            el_lm = lm[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            wr_lm = lm[mp_pose.PoseLandmark.LEFT_WRIST.value]
            shoulder = (int(sh_lm.x * w), int(sh_lm.y * h))
            elbow = (int(el_lm.x * w), int(el_lm.y * h))
            wrist = (int(wr_lm.x * w), int(wr_lm.y * h))
        else:
            sh_lm = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            el_lm = lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            wr_lm = lm[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            shoulder = (int(sh_lm.x * w), int(sh_lm.y * h))
            elbow = (int(el_lm.x * w), int(el_lm.y * h))
            wrist = (int(wr_lm.x * w), int(wr_lm.y * h))
    except Exception:
        return frame, {"status": "landmarks_missing"}

    # Draw subtle markers/lines
    cv2.circle(frame, shoulder, 4, (40, 200, 40), -1)
    cv2.circle(frame, elbow, 4, (40, 200, 40), -1)
    cv2.circle(frame, wrist, 4, (40, 200, 40), -1)
    cv2.line(frame, shoulder, elbow, (200, 180, 0), 2)
    cv2.line(frame, elbow, wrist, (200, 180, 0), 2)

    # Compute elbow angle (shoulder-elbow-wrist)
    elbow_angle = angle_calculations.calculate_angle(shoulder, elbow, wrist)
    ea = float(elbow_angle)
    ea_int = int(round(ea))

    # Smoothing (EMA)
    if _STATE["smoothed_angle"] is None:
        _STATE["smoothed_angle"] = ea
    else:
        _STATE["smoothed_angle"] = ALPHA * ea + (1 - ALPHA) * _STATE["smoothed_angle"]

    now = time.time()
    # velocity estimate (deg/sec) - helpful to filter tiny jitter
    vel = 0.0
    if _STATE["last_angle"] is not None and _STATE["last_angle_ts"] is not None:
        dt = now - _STATE["last_angle_ts"]
        if dt > 0:
            vel = (ea - _STATE["last_angle"]) / dt
    _STATE["last_angle"] = ea
    _STATE["last_angle_ts"] = now

    s = _STATE

    # Counting state machine:
    # - If not in_down and smoothed_angle <= DOWN_THRESHOLD => enter down.
    # - If in_down and smoothed_angle >= UP_THRESHOLD and cooldown passed and velocity check => count one rep.
    rep_incremented = False
    if not s["in_down"]:
        if s["smoothed_angle"] <= DOWN_THRESHOLD:
            s["in_down"] = True
    else:
        # we were down previously
        cooldown_ok = (now - s["last_rep_ts"]) * 1000.0 >= COOLDOWN_MS
        vel_ok = True if MIN_VELOCITY <= 0 else abs(vel) >= MIN_VELOCITY
        if s["smoothed_angle"] >= UP_THRESHOLD and cooldown_ok and vel_ok:
            s["rep_count"] += 1
            s["in_down"] = False
            s["last_rep_ts"] = now
            rep_incremented = True

    # status & suggestions
    if ea_int <= DOWN_THRESHOLD:
        status = "down"
    elif ea_int >= UP_THRESHOLD:
        status = "up"
    else:
        status = "mid"

    suggestions = []
    if ea_int < 60:
        suggestions.append("Controlled descent; avoid bar-bounce.")
    elif ea_int > 170:
        suggestions.append("Keep slight bend at top; avoid full lockout.")

    info = {
        "elbow_angle": ea_int,
        "smoothed_angle": int(round(s["smoothed_angle"])),
        "status": status,
        "suggestions": suggestions,
        "rep_count": int(s["rep_count"]),
        "last_rep_ts": s["last_rep_ts"]
    }

    # occasional server debug to help tuning (every ~60 frames)
    s["debug_tick"] = (s["debug_tick"] + 1) % 60
    if s["debug_tick"] == 0:
        print(f"[bench_press] smoothed={s['smoothed_angle']:.1f} angle={ea_int} vel={vel:.1f} in_down={s['in_down']} rep_count={s['rep_count']}")

    return frame, info
