import numpy as np
from scipy.stats import truncnorm
import requests
import gurobipy as gp
from gurobipy import GRB


# 🔑 API Key
with open("ETA_key.txt", "r") as file:
    FOURSQUARE_API_KEY = file.read().strip()

# ⚙️ Configuration
config = {
    "time_baseline": {
        "morning": 3.0,
        "lunch": 10.0,
        "afternoon": 2.0,
        "evening": 5.0,
        "night": 1.0,
    },
    "category_weights": {
        "university": 1.4,
        "residential": 0.5,
        "office": 1.3,
        "mall": 1.2,
        "default": 1.0,
    },
    "rating_weight": 0.25,
    "log_reviews_weight": 0.25,
    "density_weight": 0.045,
    "base_std": 2,
    "std_rating_penalty": 1.2,
    "std_review_penalty": 2.5,
    "min_wait": 1,
    "max_wait": 20,
    "sampling_iterations": 30,
}


def get_time_slot(hour):
    if 7 <= hour < 11:
        return "morning"
    elif 11 <= hour < 14:
        return "lunch"
    elif 14 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    else:
        return "night"


def truncated_normal_wait(mean, std, min_val=1, max_val=20, samples=30):
    a, b = (min_val - mean) / std, (max_val - mean) / std
    samples = truncnorm.rvs(a, b, loc=mean, scale=std, size=samples)
    return round(np.mean(samples), 2)


def fetch_foursquare_info(lat, lon):
    url = "https://api.foursquare.com/v3/places/search"
    headers = {"Authorization": FOURSQUARE_API_KEY}
    params = {
        "ll": f"{lat},{lon}",
        "radius": 100,
        "limit": 1,
        "sort": "DISTANCE",
        "fields": "rating,stats,popularity,categories",
    }

    response = requests.get(url, headers=headers, params=params)
    data = response.json()

    if not data.get("results"):
        return {
            "rating": 3.0,
            "review_count": 10,
            "density": 0.5,
            "category": "default",
        }

    result = data["results"][0]

    rating = result.get("rating", 3.0)
    review_count = result.get("stats", {}).get("total_ratings", 10)
    density = result.get("popularity", 0.5)

    category = (
        result["categories"][0]["name"].lower()
        if result.get("categories")
        else "default"
    )
    if "university" in category:
        category_key = "university"
    elif "residential" in category:
        category_key = "residential"
    elif "office" in category or "corporate" in category:
        category_key = "office"
    elif "mall" in category:
        category_key = "mall"
    else:
        category_key = "default"

    return {
        "rating": rating,
        "review_count": review_count,
        "density": density,
        "category": category_key,
    }


def estimate_wait_time(lat, lon, hour=None, config=config):
    info = fetch_foursquare_info(lat, lon)

    rating = max(info.get("rating", 4.0), 1.0)
    review_count = max(info.get("review_count", 5), 1)

    log_reviews = np.log1p(review_count)
    density = info["density"]
    category = info["category"]

    hour = hour or int(np.datetime64("now", "h").astype(int) % 24)
    time_slot = get_time_slot(hour)
    base_mean = config["time_baseline"].get(time_slot, 5)
    category_weight = config["category_weights"].get(category, 1.0)

    mean = base_mean * category_weight
    mean += (rating - 4) * config["rating_weight"]
    mean += log_reviews * config["log_reviews_weight"]
    mean += density * config["density_weight"]

    std = config["base_std"]
    std += (5 - rating) * config["std_rating_penalty"]
    std += 1 / (log_reviews + 1e-6) * config["std_review_penalty"]
    std = max(std, 0.5)

    return (
        truncated_normal_wait(
            mean,
            std,
            min_val=config["min_wait"],
            max_val=config["max_wait"],
            samples=config["sampling_iterations"],
        ),
        rating,
        density,
    )


def optimize_and_rank_cafes(
    cafes,
    max_total_time,
    min_arrival_gap,
    priority_option="Getting as fast as possible",
):
    # 사전 필터링
    filtered = []
    for cafe in cafes:
        total_time = (
            cafe["eta_start_to_cafe"] + cafe["wait_time"] + cafe["eta_cafe_to_dest"]
        )
        if total_time <= max_total_time and cafe["eta_cafe_to_dest"] >= min_arrival_gap:
            filtered.append(cafe)

    if not filtered:
        return None, []  # 조건 만족 X

    # Gurobi 모델 설정
    m = gp.Model("cafe_optimization")
    m.setParam("OutputFlag", 0)
    x = m.addVars(len(filtered), vtype=GRB.BINARY, name="x")
    m.addConstr(gp.quicksum(x[i] for i in range(len(filtered))) == 1)

    # 가중치 설정
    if priority_option == "Less crowded if possible":
        w_rating, w_wait, w_eta_dest, w_eta_start, w_density = 1.0, 1.0, 1.0, 1.0, 2.0
    elif priority_option == "Better rating if possible":
        w_rating, w_wait, w_eta_dest, w_eta_start, w_density = 2.0, 1.0, 1.0, 1.0, 1.0
    else:
        w_rating, w_wait, w_eta_dest, w_eta_start, w_density = 1.0, 2.0, 2.0, 2.0, 1.0

    # Objective 구성 및 score 계산
    objective = 0
    for i, cafe in enumerate(filtered):
        score = (
            w_rating * (cafe["rating"] / 5.0)
            - w_wait * (cafe["wait_time"] / 20.0)
            - w_eta_dest * (cafe["eta_cafe_to_dest"] / 20.0)
            - w_eta_start * (cafe["eta_start_to_cafe"] / 20.0)
            - w_density * cafe["density"]
        )
        filtered[i]["score"] = round(score, 4)
        objective += x[i] * score

    m.setObjective(objective, GRB.MAXIMIZE)
    m.optimize()

    best_cafe = None
    if m.Status == GRB.OPTIMAL:
        for i in range(len(filtered)):
            filtered[i]["is_best"] = bool(x[i].X > 0.5)
            if x[i].X > 0.5:
                filtered[i]["score"] = round(objective.getValue(), 4)
                best_cafe = filtered[i]
    else:
        print("❌ Optimization failed with status:", m.Status)

    sorted_cafes = sorted(filtered, key=lambda x: x["score"], reverse=True)
    return best_cafe, sorted_cafes
