import streamlit as st
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import requests
from geopy.distance import geodesic
from geo_modules import *
from wait_time import *

st.set_page_config(layout="centered")

st.title("Where should I get coffee today?")
st.write("☕️ Don't wait forever for coffee right in front of your office!")
st.write("Find the optimal coffee shop to get you in time ⏰")

# 📍 Step 1: Get user input locations
geolocator = Nominatim(user_agent="coffee_route_app")
st.markdown("### 📍 Step 1: Enter your locations")
start_coords, start_address = get_location_from_query("Where are you now?", geolocator)
end_coords, end_address = get_location_from_query("Where should you go?", geolocator)

if start_coords and end_coords:
    st.session_state["location_ready"] = True

if st.session_state.get("location_ready"):

    m = folium.Map(location=start_coords, zoom_start=13)
    folium.Marker(
        start_coords, tooltip="Start", icon=folium.Icon(color="green")
    ).add_to(m)
    folium.Marker(
        end_coords, tooltip="Destination", icon=folium.Icon(color="red")
    ).add_to(m)
    st_folium(m, width=700, height=500)

    if "schedule_time" not in st.session_state:
        st.session_state.schedule_time = (datetime.now() + timedelta(hours=1)).time()

    st.time_input("⏰ What time does your schedule start?", key="schedule_time")

    schedule_time = st.session_state.schedule_time

    now = datetime.now()
    schedule_datetime = datetime.combine(now.date(), schedule_time)

    # If the schedule time is before now, assume it's for tomorrow
    if schedule_datetime < now:
        schedule_datetime += timedelta(days=1)

    buffer_minutes = st.slider(
        "🚶 How many minutes early would you like to arrive?",
        min_value=0,
        max_value=30,
        value=5,
        step=1,
    )

    deadline_datetime = schedule_datetime - timedelta(minutes=buffer_minutes)

    priority = st.selectbox(
        "🔍 What is your priority?",
        [
            "Getting as fast as possible",
            "Better rating if possible",
            "Less crowded if possible",
        ],
    )

    st.write(f"🎯 You should arrive by: **{deadline_datetime.time()}**")
    # How many minutes left?
    minutes_left = round((deadline_datetime - datetime.now()).total_seconds() / 60.0, 2)
    st.write(f"⏱️ You have {minutes_left} minutes left")

    # ☕️ Step 2: Confirm and find cafes
    confirm_button = st.button("🚀 Confirm route and find nearby cafes")

    if confirm_button:
        with st.spinner("📡 Finding nearby cafes..."):
            cafes_end = find_nearby_cafes(
                end_coords[0], end_coords[1], radius_m=800, limit=5
            )
            cafes_start = find_nearby_cafes(
                start_coords[0], start_coords[1], radius_m=800, limit=5
            )

            def cafe_key(cafe):
                return (cafe["name"], round(cafe["lat"], 5), round(cafe["lon"], 5))

            seen = set()
            deduped_cafes = []
            for cafe in cafes_end + cafes_start:
                key = cafe_key(cafe)
                if key not in seen:
                    seen.add(key)
                    deduped_cafes.append(cafe)

            cafes = deduped_cafes

        if cafes:
            with st.spinner("⏱️ Calculating ETA..."):
                enriched_cafes = []

                for idx, cafe in enumerate(cafes):
                    # st.info(f"Checking ETA for **{cafe['name']}**...")
                    eta1 = get_eta_minutes(start_coords, (cafe["lat"], cafe["lon"]))
                    eta2 = get_eta_minutes((cafe["lat"], cafe["lon"]), end_coords)
                    # wait_time = 5  # TODO: Replace with wait-time proxy later
                    wait_time, rating, density, sf = estimate_wait_time_NN(
                        cafe["lat"], cafe["lon"], hour=datetime.now().hour
                    )

                    # Debug: Show the actual ETA values
                    # st.write(f"🔍 {cafe['name']}: ETA1={eta1}, ETA2={eta2}")

                    if eta1 is not None and eta2 is not None:
                        total_time = eta1 + eta2 + wait_time
                        enriched_cafes.append(
                            {
                                **cafe,
                                "eta_start_to_cafe": eta1,
                                "eta_cafe_to_dest": eta2,
                                "wait_time": wait_time,
                                "total_time": total_time,
                                "index": idx + 1,
                                "rating": rating,
                                "density": density,
                                "SF": sf,
                            }
                        )
                        # st.success(f"✅ Successfully added {cafe['name']} (Total: {total_time} min)")
                    else:
                        st.warning(f"⚠️ ETA failed for {cafe['name']} (skipping)")

                # Debug: Show final count
                st.info(
                    f"📊 Final result: {len(enriched_cafes)} cafes with valid ETAs out of {len(cafes)} found"
                )

            if enriched_cafes:
                # st.success(f"🎯 Showing {len(enriched_cafes)} optimized cafe options:")
                available_time = (
                    deadline_datetime - datetime.now()
                ).total_seconds() / 60.0

                best_cafe, sorted_cafes = optimize_and_rank_cafes(
                    enriched_cafes,
                    max_total_time=available_time,
                    min_arrival_gap=buffer_minutes,
                )
                if best_cafe:
                    st.success(
                        f"🚀 Best Option: {best_cafe['name']} (Score: {best_cafe['score']})"
                    )

                rank_emojis = ["🥇", "🥈", "🥉"] + ["🏅"] * 10  # 추가 순위 대비
                cafe_score_factors = []
                if sorted_cafes:
                    for i, cafe in enumerate(sorted_cafes):
                        emoji = rank_emojis[i] if i < len(rank_emojis) else "☕"
                        with st.expander(
                            f"{emoji} {cafe['name']} — {int(cafe['total_time'])} min total"
                        ):
                            col1, col2 = st.columns([2, 3])

                            with col1:
                                st.metric(
                                    "🚶 ETA to Cafe", f"{cafe['eta_start_to_cafe']} min"
                                )
                                # If less than 5 minutes, show as < 5 min
                                if cafe["wait_time"] <= 3:
                                    st.metric("⌛ Wait Time", "< 3 min")
                                st.metric("⌛ Wait Time", f"{cafe['wait_time']} min")
                                st.metric(
                                    "📍 ETA to Destination",
                                    f"{cafe['eta_cafe_to_dest']} min",
                                )

                            with col2:
                                st.write("### 🧠 Optimization Factors")
                                st.markdown(
                                    f"""
                                    - ⭐ **Rating**: {cafe.get('rating', 'N/A')}
                                    - 📈 **Density (lower = better)**: {round(cafe.get('density', 0), 2)}
                                    - 🧮 **Score**: {round(cafe['score'], 2)}
                                    - ⏱️ **Total Travel + Wait**: {int(cafe['eta_start_to_cafe'] + cafe['wait_time'] + cafe['eta_cafe_to_dest'])} min
                                    - ✅ **Constraints**:
                                        - Total time ≤ {int(available_time)} min
                                        - Arrive ≥ {buffer_minutes} min early
                                    - ⏱️ Wait time Score Factor: {round(float(cafe.get('SF')), 2)}
                                    """
                                )

                            st.caption(
                                f"📍 Location: ({cafe['lat']:.5f}, {cafe['lon']:.5f})"
                            )

                else:
                    # fallback best cafe
                    fallback_cafe = max(enriched_cafes, key=lambda x: x.get("score", 0))

                    total_required_time = (
                        fallback_cafe["eta_start_to_cafe"]
                        + fallback_cafe["wait_time"]
                        + fallback_cafe["eta_cafe_to_dest"]
                    )
                    required_arrival = datetime.now() + timedelta(
                        minutes=total_required_time
                    )
                    lateness = (
                        required_arrival - deadline_datetime
                    ).total_seconds() / 60.0

                    emoji = "☕"
                    with st.expander(
                        f"😂 No cafe fits your constraints, but here's the best option: {fallback_cafe['name']}"
                    ):
                        col1, col2 = st.columns([2, 3])

                        with col1:
                            st.metric(
                                "🚶 ETA to Cafe",
                                f"{fallback_cafe['eta_start_to_cafe']} min",
                            )
                            st.metric(
                                "⌛ Wait Time", f"{fallback_cafe['wait_time']} min"
                            )
                            st.metric(
                                "📍 ETA to Destination",
                                f"{fallback_cafe['eta_cafe_to_dest']} min",
                            )

                        with col2:
                            st.write("### You don't have enough time for ☕️")
                            st.markdown(
                                f"""
                                - 🧮 **Total Time Needed**: {int(total_required_time)} min  
                                - ⏱️ **You’ll be late by**: `{int(lateness)} min`
                                - ⭐ **Rating**: {fallback_cafe.get('rating', 'N/A')}
                                - 📈 **Density**: {round(fallback_cafe.get('density', 0), 2)}
                                """
                            )

                        st.caption(
                            f"📍 Location: ({fallback_cafe['lat']:.5f}, {fallback_cafe['lon']:.5f})"
                        )
                # Plot Coffee Wait Time Distribution, using plot_time_distributions
                st.markdown("### 📊 Coffee Wait Time Distribution")
                plot_time_distributions(enriched_cafes, model=model)

            else:
                st.error("All ETA calculations failed. Please try again later.")
                st.write("🔍 This could be due to:")
                st.write("- Network connectivity issues")
                st.write("- OpenRouteService API rate limits")
                st.write("- Invalid coordinates")
        else:
            st.warning("No cafes found near your destination.")
