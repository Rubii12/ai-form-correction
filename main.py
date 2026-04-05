from flask import Flask, render_template, request, jsonify
from flask_sock import Sock
import time
import threading
import json
import base64
import numpy as np

app = Flask(__name__)
sock = Sock(app)

try:
    import cv2
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except Exception as e:
    print("Warning: mediapipe/opencv not available:", e)
    MEDIAPIPE_AVAILABLE = False

from exercises import EXERCISE_MAP
from utils import angle_calculations, visualisations

LATEST_VIDEO_INFO      = {}
LATEST_VIDEO_INFO_LOCK = threading.Lock()

import random

import os
import requests
from dotenv import load_dotenv

load_dotenv()
SPOONACULAR_KEY = os.getenv("SPOONACULAR_API_KEY")

MEAL_SLOTS = [
    {"name": "Breakfast", "icon": "🥣", "pct": 0.25},
    {"name": "Lunch",     "icon": "🍱", "pct": 0.33},
    {"name": "Dinner",    "icon": "🍛", "pct": 0.30},
    {"name": "Snack",     "icon": "🥜", "pct": 0.12},
]

def suggest_meals_spoonacular(macros, tdee):
    """Fetch real Indian meals from Spoonacular matching macro targets."""
    meals = []

    for slot in MEAL_SLOTS:
        target_kcal    = int(tdee * slot["pct"])
        target_protein = int(macros.get("protein_g", 0) * slot["pct"])
        target_carbs   = int(macros.get("carbs_g",   0) * slot["pct"])
        target_fats    = int(macros.get("fats_g",    0) * slot["pct"])

        try:
            resp = requests.get(
                "https://api.spoonacular.com/recipes/complexSearch",
                params={
                    "apiKey":         SPOONACULAR_KEY,
                    "cuisine":        "Indian",
                    "number":         3,
                    "addRecipeNutrition": True,
                    "minCalories":    int(target_kcal * 0.75),
                    "maxCalories":    int(target_kcal * 1.25),
                    "minProtein":     max(0, target_protein - 10),
                    "maxProtein":     target_protein + 15,
                    "sort":           "random",
                },
                timeout=6
            )
            data = resp.json()
            results = data.get("results", [])

            if results:
                # Pick first result
                r       = results[0]
                title   = r.get("title", "Indian Dish")
                r_id    = r.get("id")
                image   = r.get("image", "")

                # Extract nutrition
                nutrients = {}
                for n in r.get("nutrition", {}).get("nutrients", []):
                    nutrients[n["name"]] = round(n["amount"])

                kcal    = nutrients.get("Calories",      target_kcal)
                protein = nutrients.get("Protein",       target_protein)
                carbs   = nutrients.get("Carbohydrates", target_carbs)
                fats    = nutrients.get("Fat",           target_fats)

                # Get ingredients summary
                ingredients = [
                    i["name"].title()
                    for i in r.get("nutrition", {}).get("ingredients", [])[:5]
                ]

                recipe_url = f"https://spoonacular.com/recipes/{title.replace(' ','-').lower()}-{r_id}"

                meals.append({
                    "name":        slot["name"],
                    "icon":        slot["icon"],
                    "title":       title,
                    "kcal":        kcal,
                    "protein":     protein,
                    "carbs":       carbs,
                    "fats":        fats,
                    "ingredients": ingredients,
                    "image":       image,
                    "url":         recipe_url,
                    "source":      "spoonacular"
                })
            else:
                # No results — fall back
                meals.append(_fallback_meal(slot, macros, tdee))

        except Exception as e:
            print(f"[spoonacular] error for {slot['name']}: {e}")
            meals.append(_fallback_meal(slot, macros, tdee))

    return meals


def _fallback_meal(slot, macros, tdee):
    """Simple fallback if API fails."""
    indian_meals = {
        "Breakfast": {"title": "Oats Upma",        "ingredients": ["Oats", "Vegetables", "Mustard seeds", "Curry leaves", "Green chilli"]},
        "Lunch":     {"title": "Dal Rice",          "ingredients": ["Toor dal", "Rice", "Ghee", "Turmeric", "Jeera"]},
        "Dinner":    {"title": "Roti Sabzi",        "ingredients": ["Whole wheat roti", "Mixed vegetables", "Paneer", "Spices", "Oil"]},
        "Snack":     {"title": "Chana Chaat",       "ingredients": ["Chickpeas", "Onion", "Tomato", "Chaat masala", "Lemon"]},
    }
    base  = indian_meals.get(slot["name"], {"title": "Indian Meal", "ingredients": []})
    kcal  = int(tdee * slot["pct"])
    return {
        "name":        slot["name"],
        "icon":        slot["icon"],
        "title":       base["title"],
        "kcal":        kcal,
        "protein":     int(macros.get("protein_g", 0) * slot["pct"]),
        "carbs":       int(macros.get("carbs_g",   0) * slot["pct"]),
        "fats":        int(macros.get("fats_g",    0) * slot["pct"]),
        "ingredients": base["ingredients"],
        "image":       "",
        "url":         "",
        "source":      "fallback"
    }

from utils import nutrition as nut
@app.route("/nutrition", methods=["GET", "POST"])
def nutrition():
    result = None
    meals  = None
    if request.method == "POST":
        try:
            weight   = float(request.form.get("weight"))
            height   = float(request.form.get("height"))
            age      = int(request.form.get("age"))
            sex      = request.form.get("sex")
            activity = request.form.get("activity")
            goal     = request.form.get("goal")
        except Exception:
            return render_template("nutrition.html", result=None, meals=None,
                                   error="Please enter valid inputs.")

        bmr    = nut.calculate_bmr(weight, height, age, sex)
        tdee   = nut.calculate_tdee(bmr, activity)
        macros = nut.macronutrients(tdee, goal)
        result = {"bmr": round(bmr), "tdee": round(tdee), **macros}

        if SPOONACULAR_KEY:
            meals = suggest_meals_spoonacular(macros, result["tdee"])
        else:
            meals = suggest_meals(macros, result["tdee"])  # old fallback

    return render_template("nutrition.html", result=result, meals=meals)


# ── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", mediapipe_available=MEDIAPIPE_AVAILABLE)

@app.route("/progress")
def progress():
    return render_template("progress.html")

@app.route("/workout")
def workout():
    return render_template("workout.html")


# ── WebSocket handler ────────────────────────────────────────────────────────

@sock.route("/ws")
def ws_workout(ws):
    """
    WebSocket handler for workout camera feed.
    Protocol:
      Client → Server: JSON { "type": "exercise", "name": "squat" }
                    or: binary JPEG frame bytes
      Server → Client: JSON { "type": "info", ...info fields... }
                    or: binary annotated JPEG frame bytes
    """
    if not MEDIAPIPE_AVAILABLE:
        ws.send(json.dumps({"type": "error", "message": "MediaPipe not available on server."}))
        return

    mp_pose     = mp.solutions.pose
    pose        = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    exercise    = "squat"

    print(f"[ws] Client connected")

    try:
        while True:
            message = ws.receive()
            if message is None:
                break

            # Text message — exercise switch command
            if isinstance(message, str):
                try:
                    data = json.loads(message)
                    if data.get("type") == "exercise":
                        exercise = data.get("name", "squat")
                        print(f"[ws] Switched to {exercise}")
                        # Reset the exercise module state
                        module = EXERCISE_MAP.get(exercise)
                        if module and hasattr(module, "reset"):
                            module.reset()
                except Exception as e:
                    print(f"[ws] JSON parse error: {e}")
                continue

            # Binary message — JPEG frame from browser
            try:
                # Decode JPEG bytes to numpy array
                jpg_arr = np.frombuffer(message, dtype=np.uint8)
                frame   = cv2.imdecode(jpg_arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                # Run MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results   = pose.process(frame_rgb)

                annotated = frame.copy()
                info      = {}

                if results.pose_landmarks:
                    module = EXERCISE_MAP.get(exercise, EXERCISE_MAP.get("squat"))
                    try:
                        annotated, info = module.process_frame(
                            annotated, results.pose_landmarks, mp_pose)
                    except Exception as e:
                        print(f"[ws] Module error: {e}")
                        annotated = frame.copy()
                        info      = {"status": "module_error"}

                # Update shared info store
                try:
                    with LATEST_VIDEO_INFO_LOCK:
                        LATEST_VIDEO_INFO.clear()
                        if info:
                            for k, v in info.items():
                                if isinstance(v, (int, float, str, bool, type(None), list, dict)):
                                    LATEST_VIDEO_INFO[k] = v
                                else:
                                    LATEST_VIDEO_INFO[k] = str(v)
                        LATEST_VIDEO_INFO["_ts"]      = time.time()
                        LATEST_VIDEO_INFO["exercise"] = exercise
                except Exception:
                    pass

                # Send annotated frame back as binary
                ret, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    ws.send(buffer.tobytes())

                # Send info as JSON
                safe_info = {}
                for k, v in info.items():
                    if isinstance(v, (int, float, str, bool, type(None), list, dict)):
                        safe_info[k] = v
                    else:
                        safe_info[k] = str(v)
                safe_info["exercise"] = exercise
                ws.send(json.dumps({"type": "info", **safe_info}))

            except Exception as e:
                print(f"[ws] Frame processing error: {e}")
                continue

    except Exception as e:
        print(f"[ws] Connection error: {e}")
    finally:
        pose.close()
        print(f"[ws] Client disconnected")


# ── REST routes ──────────────────────────────────────────────────────────────

@app.route("/video_info")
def video_info():
    with LATEST_VIDEO_INFO_LOCK:
        data = dict(LATEST_VIDEO_INFO)
    if not data:
        return jsonify({"status": "no_data"})
    return jsonify(data)


@app.route("/reset_reps", methods=["POST"])
def reset_reps():
    exercise = request.json.get("exercise", "squat")
    module   = EXERCISE_MAP.get(exercise)
    if module and hasattr(module, "reset"):
        module.reset()
        with LATEST_VIDEO_INFO_LOCK:
            LATEST_VIDEO_INFO.clear()
        return jsonify({"ok": True, "exercise": exercise})
    return jsonify({"ok": False, "reason": "unknown exercise"}), 400



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, threaded=True, use_reloader=False)