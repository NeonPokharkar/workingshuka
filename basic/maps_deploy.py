import subprocess
import time
import os

def is_docker_running():
    """Checks if the Docker engine is active."""
    try:
        subprocess.run(["docker", "info"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def start_docker_desktop():
    """Launches the Docker Desktop application."""
    print("Starting Docker Desktop...")
    subprocess.Popen(["C:\\Program Files\\Docker\\Docker\\Docker Desktop.exe"])
    # Wait for the engine to warm up
    while not is_docker_running():
        print("Waiting for Docker engine to start...")
        time.sleep(5)

def run_osrm_service():
    """Starts the OSRM India map container with an auto-restart policy."""
    if not is_docker_running():
        start_docker_desktop()

    print("Launching OSRM India Service...")
    # --restart unless-stopped ensures it starts back up if it crashes or your PC reboots
    cmd = [
        "docker", "run", "-d",
        "--name", "osrm_india",
        "--restart", "unless-stopped",
        "-p", "5000:5000",
        "-v", f"{os.getcwd().replace("\\", "/")}:/osm",
        "osrm/osrm-backend", "osrm-routed", "--algorithm", "mld", "/osm/osm/india-march.osrm"
    ]
    subprocess.run(cmd)