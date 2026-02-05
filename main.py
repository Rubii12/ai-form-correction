from flask import Flask, render_template, Response, request, jsonify
import time
import threading
import json

app = Flask(__name__)

# Try to import mediapipe and opencv; handle gracefully if not present
try:
    import cv2
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except Exception as e:
    print("Warning: mediapipe/opencv not available:", e)
    MEDIAPIPE_AVAILABLE = False

from exercises import EXERCISE_MAP
from utils import angle_calculations, visualisations

# Thread-safe holder for most recent video info (JSON-friendly)
LATEST_VIDEO_INFO = {}
LATEST_VIDEO_INFO_LOCK = threading.Lock()

# -------------------------
# Helper: simple meal suggester
# -------------------------
import random

def suggest_meals(macros, tdee):
    """
    Smarter meal suggester:
    - picks different protein/carb/fat sources per meal by shuffling candidate lists
    - calculates approximate grams per food to meet each meal's macro targets
    - returns 4 meals (Breakfast, Lunch, Dinner, Snack) with items and approximate kcal
    """
    # food DB per 100g
    foods = {
        "chicken_breast": {"label": "Chicken breast", "kcal": 165, "p":31, "f":3.6, "c":0},
        "salmon": {"label": "Salmon", "kcal":208, "p":20, "f":13, "c":0},
        "tofu": {"label": "Tofu", "kcal":76, "p":8, "f":4.8, "c":1.9},
        "egg": {"label": "Egg (whole)", "kcal":155, "p":13, "f":11, "c":1.1},
        "greek_yogurt": {"label": "Greek yogurt", "kcal":59, "p":10, "f":0.4, "c":3.6},
        "white_rice": {"label": "White rice (cooked)", "kcal":130, "p":2.7, "f":0.3, "c":28},
        "quinoa": {"label": "Quinoa (cooked)", "kcal":120, "p":4.4, "f":1.9, "c":21.3},
        "oats": {"label": "Oats (dry)", "kcal":389, "p":16.9, "f":6.9, "c":66.3},
        "sweet_potato": {"label": "Sweet potato", "kcal":86, "p":1.6, "f":0.1, "c":20.1},
        "banana": {"label": "Banana", "kcal":89, "p":1.1, "f":0.3, "c":23},
        "avocado": {"label": "Avocado", "kcal":160, "p":2, "f":15, "c":9},
        "olive_oil": {"label": "Olive oil", "kcal":884, "p":0, "f":100, "c":0},
        "almonds": {"label": "Almonds", "kcal":579, "p":21, "f":50, "c":22},
        "cottage_cheese": {"label": "Cottage cheese", "kcal":98, "p":11, "f":4.3, "c":3.4},
        "apple": {"label": "Apple", "kcal":52, "p":0.3, "f":0.2, "c":14},
    }

    # Candidate pools (balanced variety)
    proteins = ["chicken_breast", "salmon", "tofu", "egg", "greek_yogurt", "cottage_cheese"]
    carbs = ["white_rice", "quinoa", "oats", "sweet_potato", "banana", "apple"]
    fats = ["avocado", "olive_oil", "almonds"]

    # Shuffle candidate pools so each run gives different combinations
    random.shuffle(proteins)
    random.shuffle(carbs)
    random.shuffle(fats)

    # ensure pools are at least as long as number of meals by cycling if needed
    meals_list = ["Breakfast", "Lunch", "Dinner", "Snack"]
    num_meals = len(meals_list)

    # macro totals
    protein_total = macros.get("protein_g", 0) or 0
    fats_total = macros.get("fats_g", 0) or 0
    carbs_total = macros.get("carbs_g", 0) or 0

    # distribute macros across meals (slightly different split to encourage variety)
    meal_perc = {"Breakfast": 0.25, "Lunch": 0.33, "Dinner": 0.30, "Snack": 0.12}

    meal_plans = []
    # rotate index pointers so we pick different items, but keep within range
    for i, meal_name in enumerate(meals_list):
        perc = meal_perc.get(meal_name, 0.25)
        target_p = protein_total * perc
        target_f = fats_total * perc
        target_c = carbs_total * perc

        items = []

        # pick protein: rotate through proteins list
        prot_key = proteins[i % len(proteins)]
        prot_info = foods[prot_key]
        if prot_info["p"] > 0:
            grams_prot = max(20, int(round((target_p / prot_info["p"]) * 100)))  # min 20g
            items.append((prot_info["label"], f"{grams_prot} g"))
        else:
            # fallback
            items.append((foods["greek_yogurt"]["label"], "100 g"))

        # pick carb: rotate but avoid using same as protein if weird overlap
        carb_key = carbs[i % len(carbs)]
        carb_info = foods[carb_key]
        if target_c > 8 and carb_info["c"] > 0:
            grams_c = int(round((target_c / carb_info["c"]) * 100))
            grams_c = max(30, grams_c)
            items.append((carb_info["label"], f"{grams_c} g"))
        else:
            # light carb
            items.append((foods["banana"]["label"], "1 medium"))

        # pick fat: rotate and for snacks prefer nuts/oil
        fat_key = fats[i % len(fats)]
        fat_info = foods[fat_key]
        if target_f >= 4 and fat_info["f"] > 0:
            grams_f = int(round((target_f / fat_info["f"]) * 100))
            grams_f = max(8, grams_f)
            items.append((fat_info["label"], f"{grams_f} g"))
        else:
            # small flavoring
            items.append((foods["olive_oil"]["label"], "5 g (1 tsp)"))

        # Sometimes add a small extra (fruit or yogurt) to make meals different and balanced
        if meal_name == "Breakfast":
            items.append((foods["oats"]["label"], "30 g"))
        elif meal_name == "Snack":
            items.append((foods["almonds"]["label"], "20 g"))
        elif meal_name == "Lunch" and carb_key == "white_rice":
            items.append((foods["apple"]["label"], "1 medium"))

        # compute approximate kcal for meal
        kcal = 0
        for label, qty in items:
            # resolve key
            key = next((k for k, v in foods.items() if v["label"] == label), None)
            if key:
                if "g" in qty:
                    q = int(qty.split()[0])
                    kcal += foods[key]["kcal"] * (q / 100.0)
                else:
                    # approximate single fruit/serving
                    if qty.strip().lower() in ("1 medium", "1 serving", "1 piece"):
                        kcal += foods[key]["kcal"] * 0.5
                    elif "tsp" in qty:
                        kcal += foods[key]["kcal"] * 0.05
                    else:
                        kcal += foods[key]["kcal"] * 0.3
            else:
                kcal += 50

        # finalize meal
        meal_plans.append({
            "name": meal_name,
            "items": items,
            "kcal": int(round(kcal))
        })

    # Shuffle the order of protein/carbs/fats pools for next call side-effect already applied;
    # also shuffle the generated meals list slightly to vary presentation while keeping breakfast first
    # Keep breakfast at index 0, shuffle others
    rest = meal_plans[1:]
    random.shuffle(rest)
    meal_plans = [meal_plans[0]] + rest

    return meal_plans

# -------------------------
# Routes
# -------------------------
# Simple homepage
@app.route("/")
def index():
    return render_template("index.html", mediapipe_available=MEDIAPIPE_AVAILABLE)

# Progress page
@app.route("/progress")
def progress():
    return render_template("progress.html")

@app.route("/workout")
def workout():
    return render_template("workout.html")

# Camera feed route (if mediapipe available)
def gen_frames_for_exercise(exercise_name="squat"):
    if not MEDIAPIPE_AVAILABLE:
        msg_frame = visualisations.make_text_frame("MediaPipe or OpenCV not installed.\nInstall to use camera features.")
        while True:
            ret, jpg = cv2.imencode('.jpg', msg_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')
            time.sleep(0.2)

    cap = cv2.VideoCapture(0)
    # small safety
    if not cap.isOpened():
        msg_frame = visualisations.make_text_frame("Could not open webcam.")
        while True:
            ret, jpg = cv2.imencode('.jpg', msg_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')
            time.sleep(0.2)

    # initialize mediapipe pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            # call exercise-specific detector (returns annotated frame and info)
            annotated = frame.copy()
            info = {}
            if results.pose_landmarks:
                module = EXERCISE_MAP.get(exercise_name, EXERCISE_MAP.get("squat"))
                try:
                    annotated, info = module.process_frame(annotated, results.pose_landmarks, mp_pose)
                except Exception:
                    annotated = frame.copy()
                    info = {"status": "module_error"}

            # --- update shared latest info (thread-safe) ---
            try:
                with LATEST_VIDEO_INFO_LOCK:
                    LATEST_VIDEO_INFO.clear()
                    if info:
                        for k, v in info.items():
                            # only store JSON-friendly primitives; convert others to strings
                            if isinstance(v, (int, float, str, bool, type(None))):
                                LATEST_VIDEO_INFO[k] = v
                            else:
                                try:
                                    LATEST_VIDEO_INFO[k] = json.dumps(v)
                                except Exception:
                                    LATEST_VIDEO_INFO[k] = str(v)
                    # small metadata
                    LATEST_VIDEO_INFO["_ts"] = time.time()
                    LATEST_VIDEO_INFO["exercise"] = exercise_name
            except Exception:
                pass

            # NOTE: comment out heavy in-frame drawing so UI info is outside the frame.
            # visualisations.draw_info(annotated, info)

            ret, buffer = cv2.imencode('.jpg', annotated)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()
        pose.close()

@app.route("/video_feed")
def video_feed():
    # exercise selection
    exercise = request.args.get("exercise", "squat")
    return Response(gen_frames_for_exercise(exercise),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# New route: return latest video info as JSON for the popup UI
@app.route("/video_info")
def video_info():
    with LATEST_VIDEO_INFO_LOCK:
        data = dict(LATEST_VIDEO_INFO)  # copy
    if not data:
        return jsonify({"status": "no_data"})
    return jsonify(data)

from utils import nutrition as nut

@app.route("/nutrition", methods=["GET", "POST"])
def nutrition():
    result = None
    meals = None
    if request.method == "POST":
        try:
            weight = float(request.form.get("weight"))
            height = float(request.form.get("height"))
            age = int(request.form.get("age"))
            sex = request.form.get("sex")
            activity = request.form.get("activity")
            goal = request.form.get("goal")
        except Exception:
            return render_template("nutrition.html", result=None, meals=None, error="Please enter valid inputs.")

        bmr = nut.calculate_bmr(weight, height, age, sex)
        tdee = nut.calculate_tdee(bmr, activity)
        macros = nut.macronutrients(tdee, goal)

        result = {"bmr": round(bmr), "tdee": round(tdee), **macros}

        # generate meal suggestions (approximate)
        try:
            meals = suggest_meals(macros, result["tdee"])
        except Exception:
            meals = None

    return render_template("nutrition.html", result=result, meals=meals)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

