# Basic Producer Execution Script

# Executes the producer service with default simulation parameters
# using the ALMA antenna configuration.

# Configuration paths
ANTENNA_CONFIG="./antenna_configs/skamid.cfg"
SIMULATION_CONFIG="./configs/simulation/ska-mid-band-02.json"

# Service path
PRODUCER_SERVICE="./services/producer_service.py"

# Execute producer service
python3 "$PRODUCER_SERVICE" "$ANTENNA_CONFIG" --simulation-config "$SIMULATION_CONFIG"
