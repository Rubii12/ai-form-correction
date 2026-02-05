import math

def calculate_angle(a, b, c):
    """Return angle at point b formed by points a-b-c (in degrees). a,b,c are (x,y) tuples."""
    (ax, ay), (bx, by), (cx, cy) = a, b, c
    v1 = (ax - bx, ay - by)
    v2 = (cx - bx, cy - by)
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])
    if mag1*mag2 == 0:
        return 0.0
    cosang = max(-1.0, min(1.0, dot / (mag1*mag2)))
    angle = math.degrees(math.acos(cosang))
    return angle
