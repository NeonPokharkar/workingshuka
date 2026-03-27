from geopy.geocoders import Nominatim
from langchain.tools import ToolException, tool
import requests

import static.constants as constants

## Maps Setup

geolocator = Nominatim(user_agent="india_map_agent")

@tool
def get_routing_data(coordinates: str) -> str:
    """
        Calculates driving distance and time. 
        Input: 'start_lon,start_lat;end_lon,end_lat'
        """
    try:
        # OSRM local server URL (assumes port 5000 is mapped in Docker)
        url = f"{constants.MAP_DRIVING_BASE_URL}/{coordinates}?overview=false"
    
        response = requests.get(url).json()
        if response.get("code") == "Ok":
            route = response["routes"][0]
            dist_km = route["distance"] / 1000
            time_min = route["duration"] / 60
            return f"Distance: {dist_km:.2f} km. Travel time: {time_min:.1f} minutes."
        raise ToolException(f"OSRM Error: {response.get('message', 'Unknown error')}")
    except Exception as e:
        raise ToolException(f"Error calculating route: {str(e)}")

@tool
def identify_place_tool(location_name: str) -> str:
    """Finds the coordinates for a place name in India."""
    try:
        # OSRM requires (Longitude, Latitude) order
        location = geolocator.geocode(location_name)
        if location:
            return f"{location.longitude},{location.latitude}"
        raise ToolException("Place not found.")
    except Exception as e:
        raise ToolException(f"Error identifying place: {str(e)}")

@tool
def get_osrm_directions(coordinates: str) -> str:
    """
    Provides turn-by-turn directions between two points in India using OSRM.
    Input: 'start_lon,start_lat;end_lon,end_lat'
    """
    # OSRM expects coordinates in {longitude},{latitude} format
    # Example URL: http://localhost:5000/route/v1/driving/lon,lat;lon,lat?steps=true
    
    url = f"{constants.MAP_DRIVING_BASE_URL}/{coordinates}"
    params = {
        "steps": "true",        # Returns turn-by-turn maneuvers
        "geometries": "geojson",
        "overview": "false"
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        if data.get("code") != "Ok":
            return f"OSRM Error: {data.get('message', 'Unknown error')}"

        # Extracting the maneuvers from the first route
        route = data["routes"][0]
        instructions = []
        
        # OSRM routes are divided into 'legs' (between waypoints)
        for leg in route["legs"]:
            for step in leg["steps"]:
                maneuver = step["maneuver"]
                street = step.get("name", "unnamed road")
                dist = int(step["distance"])
                
                # Format: "Turn right onto [Street] and go [X] meters"
                action = f"{maneuver['type']} {maneuver.get('modifier', '')}".strip()
                instr = f"{action.capitalize()} onto {street} ({dist}m)"
                instructions.append(instr)

        summary = f"Total distance: {int(route['distance'])}m, Duration: {int(route['duration']/60)} mins."
        return f"{summary}\n\nSteps:\n" + "\n".join(instructions)

    except Exception as e:
        raise ToolException(f"Error fetching directions: {str(e)}")
    

tool_list = [get_routing_data, identify_place_tool, get_osrm_directions]