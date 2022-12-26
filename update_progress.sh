#!/bin/bash

# Set the path to the source file and the target file
source_file="experiment.out"
target_file="experiment.status.txt"

# Check if the target file exists
if [ ! -f "$target_file" ]
then
    # Create the target file if it doesn't exist
    touch "$target_file"
fi

# Check if the source file has been modified
while true
do
    # Get the last line of the source file
    last_line=$(tail -n 1 "$source_file")

    # Check if the last line is different from the current contents of the target file
    if [ "$last_line" != "$(cat "$target_file")" ]
    then
        # Overwrite the contents of the target file with the last line of the source file
        echo "$last_line" > "$target_file"
    fi

    # Sleep for 1 second before checking for updates again
    sleep 5
done
