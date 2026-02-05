# utils/nutrition.py

def calculate_bmr(weight_kg, height_cm, age, sex):
    if sex.lower() == "male":
        return 88.36 + (13.4 * weight_kg) + (4.8 * height_cm) - (5.7 * age)
    else:
        return 447.6 + (9.2 * weight_kg) + (3.1 * height_cm) - (4.3 * age)

def calculate_tdee(bmr, activity_level):
    factors = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very_active": 1.9
    }
    return bmr * factors.get(activity_level, 1.2)

def macronutrients(tdee, goal="maintain"):
    # default macros as % of calories
    if goal == "muscle_gain":
        protein = 0.3 * tdee / 4  # grams
        fats = 0.25 * tdee / 9
        carbs = 0.45 * tdee / 4
    elif goal == "fat_loss":
        protein = 0.35 * tdee / 4
        fats = 0.25 * tdee / 9
        carbs = 0.4 * tdee / 4
    else:  # maintain
        protein = 0.3 * tdee / 4
        fats = 0.3 * tdee / 9
        carbs = 0.4 * tdee / 4
    return {"protein_g": round(protein), "fats_g": round(fats), "carbs_g": round(carbs)}
