import subprocess
import os

# Configuration
PBF_FILE = "india-march.osm.pbf"
BASE_NAME = "india-march"
OSRM_IMAGE = "osrm/osrm-backend"
# Get current directory for Docker mounting
DATA_DIR = os.getcwd().replace("\\", "/")

# --- RESOURCE LIMITS ---
# Adjust these based on your PC's total specs
CPU_LIMIT = "8.0"      # Hard cap: Docker cannot use more than 4 cores
THREAD_LIMIT = "8"     # Soft cap: Tells OSRM to only spawn 4 threads
RAM_LIMIT = "28g"      # Safety cap: Prevents Docker from eating all 16GB

def run_osrm_command(step_name, command_args):
    print(f"--- Starting {step_name} ---")
    # -v mounts your current folder to /data inside the container
    docker_base = ["docker", "run", "--rm", "-t", "--cpus", CPU_LIMIT, "--memory", RAM_LIMIT, "-v", f"{DATA_DIR}:/osm", OSRM_IMAGE]
    full_cmd = docker_base + command_args
    
    result = subprocess.run(full_cmd)
    if result.returncode != 0:
        print(f"❌ {step_name} failed. Check your RAM and Docker settings.")
        exit(1)
    print(f"✅ {step_name} completed successfully.\n")

if __name__ == "__main__":
    # 1. EXTRACT: Pulls the road network based on 'car' profile
    # run_osrm_command("Extract", ["osrm-extract", "-t", THREAD_LIMIT, "-p", "/opt/car.lua", f"/osm/osm/{PBF_FILE}"])

    # 2. PARTITION: Divides the map into smaller 'cells' for faster math
    run_osrm_command("Partition", ["osrm-partition", "--threads", THREAD_LIMIT, f"/osm/osm/{BASE_NAME}.osrm"])

    # 3. CUSTOMIZE: Calculates travel times and weights
    run_osrm_command("Customize", ["osrm-customize", "--threads", THREAD_LIMIT, f"/osm/osm/{BASE_NAME}.osrm"])

    print("🎉 All done! Your India map is ready for the routing server.")
