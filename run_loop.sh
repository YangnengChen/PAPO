#!/bin/bash

TARGET_SCRIPT="/home/hector6/PAPO/examples/papo_dapo/qwen2_5_vl_7b_dapo_ours_copy.sh"
DELAY=3

PID_TO_WAIT_FOR="3211924" 

WAIT_INTERVAL=10

if [ -n "$PID_TO_WAIT_FOR" ]; then
    echo "-----------------------------------------------------"
    echo "Pre-run check: Waiting for PID $PID_TO_WAIT_FOR to finish..."

    while [ -d "/proc/$PID_TO_WAIT_FOR" ]; do
        echo "Process $PID_TO_WAIT_FOR is still running. Waiting $WAIT_INTERVAL seconds..."
        sleep $WAIT_INTERVAL
    done
    
    echo "Process $PID_TO_WAIT_FOR has finished (or does not exist)."
    echo "-----------------------------------------------------"
else
    echo "No pre-wait PID specified. Starting immediately."
fi

echo "Starting auto-retry wrapper for: $TARGET_SCRIPT"
echo "Will retry every $DELAY seconds upon failure."
echo "-----------------------------------------------------"

while ! bash "$TARGET_SCRIPT"; do
    EXIT_CODE=$?
    
    echo "-----------------------------------------------------"
    echo "Script $TARGET_SCRIPT failed with exit code $EXIT_CODE."
    echo "Retrying in $DELAY seconds..."
    echo "-----------------------------------------------------"
    
    sleep $DELAY
done

echo "-----------------------------------------------------"
echo "$TARGET_SCRIPT completed successfully."
echo "Wrapper script finished."
echo "-----------------------------------------------------"