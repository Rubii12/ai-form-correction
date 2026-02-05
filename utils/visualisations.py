import cv2
import numpy as np

def make_text_frame(text, width=640, height=480):
    img = np.zeros((height, width, 3), dtype=np.uint8) + 40
    y0 = 40
    for i, line in enumerate(text.splitlines()):
        cv2.putText(img, line, (20, y0 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    return img

def draw_info(frame, info):
    """
    Reduced drawing: avoid drawing a large black box.
    If you want minimal in-frame debug text, uncomment the small-text section below.
    """
    # Do nothing by default to avoid the in-frame black box. The UI will show info externally.
    return

    # --- Optional minimal overlay (uncomment to enable) ---
    h, w = frame.shape[:2]
    lines = []
    if not info:
        lines = ["No data"]
    else:
        if "knee_angle" in info:
            lines.append(f"Knee angle: {info['knee_angle']}°")
        if "status" in info:
            lines.append(f"Status: {info['status']}")
        if "suggestions" in info and info['suggestions']:
            # only show the first suggestion short
            lines.append(info['suggestions'][0])
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (12, 26 + i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
