import numpy as np
from scipy.stats import truncnorm
import requests
import gurobipy as gp
from gurobipy import GRB
from train_NN_model import GeneralizedSigmoid
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import streamlit as st

# 🔑 API Key
with open("ETA_key.txt", "r") as file:
    FOURSQUARE_API_KEY = file.read().strip()

# ⚙️ Configuration
statistical_config = {
    "time_baseline": {
        "morning": 3.0,
        "lunch": 10.0,
        "afternoon": 2.0,
        "evening": 5.0,
        "night": 1.0,
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
            "rating": 5.0,
            "density": 0.9,
        }

    result = data["results"][0]

    rating = result.get("rating", 5.0)
    density = result.get("popularity", 0.9)

    return {
        "rating": rating,
        "density": density,
    }


def estimate_wait_time_statmodel(lat, lon, hour=None, config=statistical_config):
    info = fetch_foursquare_info(lat, lon)

    rating = max(info.get("rating", 4.0), 1.0)
    review_count = max(info.get("review_count", 5), 1)

    log_reviews = np.log1p(review_count)
    density = info["density"]

    hour = hour or int(np.datetime64("now", "h").astype(int) % 24)
    time_slot = get_time_slot(hour)
    base_mean = config["time_baseline"].get(time_slot, 5)

    mean = base_mean
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

    filtered = []
    for cafe in cafes:
        total_time = (
            cafe["eta_start_to_cafe"] + cafe["wait_time"] + cafe["eta_cafe_to_dest"]
        )
        if total_time <= max_total_time and cafe["eta_cafe_to_dest"] >= min_arrival_gap:
            filtered.append(cafe)

    if not filtered:
        return None, []

    m = gp.Model("cafe_optimization")
    m.setParam("OutputFlag", 0)
    x = m.addVars(len(filtered), vtype=GRB.BINARY, name="x")
    m.addConstr(gp.quicksum(x[i] for i in range(len(filtered))) == 1)

    if "crowded" in priority_option.lower():
        w_rating, w_wait, w_eta_dest, w_eta_start, w_density = 1.0, 1.0, 1.0, 1.0, 4.0
    elif "rating" in priority_option.lower():
        w_rating, w_wait, w_eta_dest, w_eta_start, w_density = 4.0, 1.0, 1.0, 1.0, 1.0
    else:
        w_rating, w_wait, w_eta_dest, w_eta_start, w_density = 1.0, 2.0, 2.0, 2.0, 1.0

    objective = 0
    for i, cafe in enumerate(filtered):
        score = (
            w_rating * (cafe["rating"])
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


### Wait time based on Baseline Model - Sigmoid
model_config = {
    "time_baseline": {
        "morning": 7.0,
        "lunch": 10.0,
        "afternoon": 5.0,
        "evening": 3.0,
        "night": 1.0,
    },
    "PEAK_TIMES": {
        "lunch": 12,
        "evening": 18,
    },
}


def proximity_weight(hour, peak_hours):
    min_diff = min(abs(hour - ph) for ph in peak_hours)
    return np.exp(-0.3 * min_diff)


def compute_score_factor(info, hour, config=model_config):
    time_slot = get_time_slot(hour)
    time_baseline = config["time_baseline"].get(time_slot, 5.0)
    peak_hours = list(config["PEAK_TIMES"].values())

    rating = max(info.get("rating", 4.0), 1.0)
    density = info.get("density", 0.5)
    proximity = proximity_weight(hour, peak_hours)

    stretched_density = (density - 0.9) * 10
    stretched_density = max(0.0, min(stretched_density, 1.0))
    curved_proximity = 1.0 / (1.0 + np.exp(-5 * (proximity - 0.5)))
    inverse_rating = 1.0 / rating

    score = (
        2.0 * stretched_density
        + 3.0 * curved_proximity
        + 1.5 * inverse_rating
        + 2.0 * (time_baseline / 10.0)
    )

    return score, rating, density, proximity


def logistic_sigmoid(x, L=10.83, k=5.00, x0=4.45, offset=3.00):
    return L / (1 + np.exp(-k * (x - x0))) + offset


def predict_wait_time(score, L=10.83, k=5.00, x0=4.45, offset=3.00):
    return logistic_sigmoid(score, L=L, k=k, x0=x0, offset=offset)


# 3. Wait Time
def estimate_wait_time_curvemodel(lat, lon, hour=None, config=model_config):
    info = fetch_foursquare_info(lat, lon)
    hour = hour or int(np.datetime64("now", "h").astype(int) % 24)

    # sf, rating, density, review_count = compute_score_factor(info, hour, config)
    sf, rating, density, proximity = compute_score_factor(info, hour, config)
    wait_time = round(predict_wait_time(sf), 2)

    return (wait_time, rating, density, sf)


#### From here: NN model
model_config = {
    "PEAK_TIMES": {
        "lunch": 12,
        "evening": 18,
    },
}


def transform_input_factor(info, hour, config):

    output = np.zeros(8)

    peak_hours = list(config["PEAK_TIMES"].values())

    rating = max(info.get("rating", 4.0), 1.0)
    density = info.get("density", 0.5)
    proximity = proximity_weight(hour, peak_hours)
    stretched_density = (density - 0.9) * 10

    stretched_density = max(0.0, min(stretched_density, 1.0))
    curved_proximity = np.log1p(5 * proximity)
    inverse_rating = 1.0 / rating

    time_slot = get_time_slot(hour)
    if time_slot == "afternoon":
        time_onehot_index = 0
    elif time_slot == "evening":
        time_onehot_index = 1
    elif time_slot == "lunch":
        time_onehot_index = 2
    elif time_slot == "morning":
        time_onehot_index = 3
    else:
        time_onehot_index = 4

    output[0] = inverse_rating
    output[1] = stretched_density
    output[2] = curved_proximity
    output[time_onehot_index + 3] = 1.0  # One-hot encoding for time slot

    return output, rating, density


model = GeneralizedSigmoid(input_dim=8)
model.load_state_dict(torch.load("sigmoid_model.pt"))


def predict_wait_time_NN(input_data, model=model):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(input_data).float().unsqueeze(0)  # (1, input_dim)

        # z = w^T x + b
        z = model.linear(x_tensor)  # (1, 1)
        y = model.forward(x_tensor)  # predicted wait time

        return y.item(), z.item()


def estimate_wait_time_NN(lat, lon, hour=None, config=model_config):
    info = fetch_foursquare_info(lat, lon)
    hour = hour or int(np.datetime64("now", "h").astype(int) % 24)

    # sf, rating, density, review_count = compute_score_factor(info, hour, config)
    input_data, rating, density = transform_input_factor(info, hour, config)
    wait_time = predict_wait_time_NN(input_data)[0]
    score_factor = predict_wait_time_NN(input_data)[1]
    return round(wait_time, 2), rating, density, score_factor


def plot_time_distributions(sorted_cafes, model):
    with torch.no_grad():
        z_range = np.linspace(-1, 2, 300)
        z_tensor = torch.tensor(z_range).unsqueeze(1).float()

        y_smooth = (
            model.L / (1 + torch.exp(-model.k * (z_tensor - model.x0))) + model.offset
        )

        z_data = [i["SF"] for i in sorted_cafes]
        y_data = [i["wait_time"] for i in sorted_cafes]
        names = [i["name"] for i in sorted_cafes]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            z_range,
            y_smooth.numpy(),
            color="red",
            linewidth=2,
            label="Fitted Sigmoid Curve",
        )
        ax.scatter(z_data, y_data, color="blue", label="Actual Wait Times", alpha=0.6)

        for i, name in enumerate(names):
            ax.annotate(name, (z_data[i], y_data[i]), fontsize=8, alpha=0.7)

        ax.set_xlabel("Linear Score (z = wᵀx + b) (Higher, more likely to bw crowded)")
        ax.set_ylabel("Predicted Wait Time based on model")
        ax.set_title("Generalized Sigmoid Fit with Data")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
