import numpy as np
from scipy.stats import truncnorm
import requests

# 🔑 API Key
with open("ETA_key.txt", "r") as file:
    FOURSQUARE_API_KEY = file.read().strip()

# ⚙️ Configuration
config = {
    "time_baseline": {
        "morning": 2.5,
        "lunch": 11.0,
        "afternoon": 4.0,
        "evening": 6.0,
        "night": 2.0,
    },
    "category_weights": {
        "university": 1.4,
        "residential": 0.6,
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
        return {"rating": 4.0, "review_count": 10, "density": 50, "category": "default"}

    result = data["results"][0]

    rating = result.get("rating", 4.0)
    review_count = result.get("stats", {}).get("total_ratings", 10)
    density = result.get("popularity", 50)

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

    return truncated_normal_wait(
        mean,
        std,
        min_val=config["min_wait"],
        max_val=config["max_wait"],
        samples=config["sampling_iterations"],
    )
