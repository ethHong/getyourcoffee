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

# 둘 다 좌표가 선택된 경우
if start_coords and end_coords:
    # 🗺 초기 지도 표시
    m = folium.Map(location=start_coords, zoom_start=13)
    folium.Marker(
        start_coords, tooltip="Start", icon=folium.Icon(color="green")
    ).add_to(m)
    folium.Marker(
        end_coords, tooltip="Destination", icon=folium.Icon(color="red")
    ).add_to(m)
    st_folium(m, width=700, height=500)

    # ☕️ Step 2: Confirm and find cafes
    confirm_button = st.button("🚀 Confirm route and find nearby cafes")

    if confirm_button:
        with st.spinner("📡 Finding nearby cafes..."):
            cafes = find_nearby_cafes(
                end_coords[0], end_coords[1], radius_m=800, limit=5
            )

        if cafes:
            with st.spinner("⏱️ Calculating ETA..."):
                enriched_cafes = []

                for idx, cafe in enumerate(cafes):
                    # st.info(f"Checking ETA for **{cafe['name']}**...")
                    eta1 = get_eta_minutes(start_coords, (cafe["lat"], cafe["lon"]))
                    eta2 = get_eta_minutes((cafe["lat"], cafe["lon"]), end_coords)
                    # wait_time = 5  # TODO: Replace with wait-time proxy later
                    wait_time = estimate_wait_time(cafe["lat"], cafe["lon"])

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
                for cafe in enriched_cafes:
                    with st.expander(
                        f"☕ {cafe['index']}. {cafe['name']} — {int(cafe['total_time'])} min total"
                    ):
                        col1, col2, col3 = st.columns(3)

                        col1.metric("ETA to Cafe", f"{cafe['eta_start_to_cafe']} min")
                        col2.metric("Wait Time", f"{cafe['wait_time']} min")
                        col3.metric(
                            "ETA to Destination", f"{cafe['eta_cafe_to_dest']} min"
                        )

                        st.caption(
                            f"📍 Location: ({cafe['lat']:.5f}, {cafe['lon']:.5f})"
                        )

            else:
                st.error("All ETA calculations failed. Please try again later.")
                st.write("🔍 This could be due to:")
                st.write("- Network connectivity issues")
                st.write("- OpenRouteService API rate limits")
                st.write("- Invalid coordinates")
        else:
            st.warning("No cafes found near your destination.")
