import streamlit as st
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import requests
from geopy.distance import geodesic
import openrouteservice

# Read api_key.txt
# with open("api_key.txt", "r") as file:
#    ORS_API_KEY = file.read().strip()

ORS_API_KEY = st.secrets.get("ORS_API_KEY", None)
ors_client = openrouteservice.Client(key=ORS_API_KEY)


# Query location based on user input text through Nominatim API
def get_location_from_query(
    label: str, geolocator=Nominatim(user_agent="coffee_route_app")
):
    query = st.text_input(f"{label} - Type an address or place name", "")

    search_button = st.button(f"🔍 Search {label}")

    if f"{label}_result" not in st.session_state:
        st.session_state[f"{label}_result"] = None
        st.session_state[f"{label}_query"] = None

    if search_button:
        try:
            results = geolocator.geocode(query, exactly_one=False, limit=5)
            if results:
                st.session_state[f"{label}_result"] = results
                st.session_state[f"{label}_query"] = query
            else:
                st.warning("No results found.")
        except Exception as e:
            st.error(
                f"⚠️ Service temprarily anavailable due to high usage! Please try later : {e}"
            )
            return None, None

    results = st.session_state.get(f"{label}_result")
    if results:
        try:
            options = [
                f"{r.address} ({r.latitude:.5f}, {r.longitude:.5f})" for r in results
            ]
            selected = st.selectbox(
                f"{label} - Select a location from results", options
            )
            selected_idx = options.index(selected)
            selected_location = results[selected_idx]

            if (
                selected_location.latitude is None
                or selected_location.longitude is None
            ):
                st.error("Selected location does not have valid coordinates.")
                return None, None

            coords = (selected_location.latitude, selected_location.longitude)
            st.success(f"{label} selected: {selected_location.address}")
            return coords, selected_location.address
        except Exception as e:
            st.error(f"⚠️ Error selecting location: {e}")
            return None, None

    return None, None


def find_nearby_cafes(lat, lon, radius_m=1000, limit=5):
    # Multiple Overpass API servers to try
    overpass_urls = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.openstreetmap.ru/api/interpreter",
    ]

    query = f"""
    [out:json][timeout:15];
    node["amenity"="cafe"](around:{radius_m},{lat},{lon});
    out body;
    """

    st.write(f"🔍 Searching for cafes within {radius_m}m of ({lat:.4f}, {lon:.4f})")

    # Try each Overpass server
    for i, overpass_url in enumerate(overpass_urls):
        try:
            # st.write(f"🌐 Trying server {i+1}/3: {overpass_url.split('//')[1].split('/')[0]}")
            response = requests.get(overpass_url, params={"data": query}, timeout=10)

            if response.status_code == 200 and response.text.strip():
                data = response.json()
                cafes = data.get("elements", [])

                if cafes:
                    # st.success(f"✅ Found {len(cafes)} cafes from {overpass_url.split('//')[1].split('/')[0]}")
                    break
                else:
                    st.warning(f"⚠️ Server responded but found no cafes")

        except requests.exceptions.Timeout:
            st.warning(f"⏱️ Server {i+1} timed out, trying next...")
            continue
        except requests.exceptions.RequestException as e:
            st.warning(f"🌐 Server {i+1} error: {e}")
            continue
        except Exception as e:
            st.warning(f"🚨 Server {i+1} unexpected error: {e}")
            continue

    # If no servers worked, use mock data for demonstration
    if not cafes:
        st.warning("🔄 All Overpass servers failed. Using demo cafes for testing...")

        # Generate some realistic mock cafes near the location
        mock_cafes = [
            {
                "lat": lat + 0.001,
                "lon": lon + 0.001,
                "tags": {"name": "Demo Coffee Shop"},
            },
            {"lat": lat - 0.001, "lon": lon + 0.002, "tags": {"name": "Test Cafe"}},
            {
                "lat": lat + 0.002,
                "lon": lon - 0.001,
                "tags": {"name": "Sample Roasters"},
            },
            {
                "lat": lat - 0.0015,
                "lon": lon - 0.0015,
                "tags": {"name": "Mock Espresso Bar"},
            },
            {
                "lat": lat + 0.0025,
                "lon": lon + 0.0005,
                "tags": {"name": "Demo Brew House"},
            },
        ]

        cafes = mock_cafes[:limit]
        st.info(f"📍 Using {len(cafes)} demo cafes for testing")

    if not cafes:
        st.error("No cafes found and fallback failed. Please try a different location.")
        return []

    # Process the cafes (whether real or mock)
    try:
        # Limit results and add distance
        cafes = cafes[:limit]
        for cafe in cafes:
            cafe["distance"] = geodesic((lat, lon), (cafe["lat"], cafe["lon"])).meters

        cafes = sorted(cafes, key=lambda c: c["distance"])[:limit]

        result = [
            {
                "name": cafe.get("tags", {}).get("name", "Unnamed Cafe"),
                "lat": cafe["lat"],
                "lon": cafe["lon"],
            }
            for cafe in cafes
        ]

        st.success(f"✅ Processed {len(result)} cafes successfully")
        return result

    except Exception as e:
        st.error(f"🚨 Error processing cafe data: {e}")
        return []


def get_eta_minutes(start_coords, end_coords, mode="foot-walking"):
    try:
        # Debug: Show coordinates being used
        # st.write(f"🛣️ Calculating route from {start_coords} to {end_coords}")

        route = ors_client.directions(
            coordinates=[
                (start_coords[1], start_coords[0]),  # lon, lat format for ORS
                (end_coords[1], end_coords[0]),
            ],
            profile=mode,
            format="geojson",
        )
        duration_sec = route["features"][0]["properties"]["summary"]["duration"]
        eta_minutes = round(duration_sec / 60, 1)
        # st.write(f"✅ Route calculated: {eta_minutes} minutes")
        return eta_minutes
    except Exception as e:
        st.error(f"🚨 ORS route error: {e}")
        st.write(f"📍 Failed coordinates: {start_coords} → {end_coords}")
        return None
