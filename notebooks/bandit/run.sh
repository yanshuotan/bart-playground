#!/usr/bin/env bash

# Create or append to a log file in your working directory
LOGFILE="sh_log/run_$(date +%Y%m%dT%H%M%S).log"

# Redirect stdout and stderr to both console and $LOGFILE
exec > >(tee -ia "$LOGFILE")
exec 2> >(tee -ia "$LOGFILE" >&2)

# Continous datasets
python compare.py "Linear"
python compare.py "Friedman"
python compare.py "LinFriedman"
python compare.py "Friedman2"
python compare.py "Friedman3"

# Classification datasets
python compare.py "Magic"
python compare.py "Shuttle"
python compare.py "Adult"
# python compare.py "Covertype"
# python compare.py "MNIST"
python compare.py "Mushroom"

# At the end of run_all.sh
echo "All tasks complete. Shutting down in 2 minutes…" 
sudo shutdown -h +2  # Shutdown after 2 minutes
