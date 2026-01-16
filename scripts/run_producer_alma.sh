# Basic Producer Execution Script

# Executes the producer service with default simulation parameters
# using the ALMA antenna configuration.

# Configuration paths
ANTENNA_CONFIG="./antenna_configs/alma.cycle10.1.cfg"
SIMULATION_CONFIG="./configs/simulation/alma-band-02.json"

# Service path
PRODUCER_SERVICE="./services/producer_service.py"

# Execute producer service
python3 "$PRODUCER_SERVICE" "$ANTENNA_CONFIG" --simulation-config "$SIMULATION_CONFIG"
