#!/bin/bash

# Set the path to the source file and the target file
source_file="/path/to/source/file"
target_file="/path/to/target/file"

# Check if the source file has been modified
while true
do
    # Get the last line of the source file
    last_line=$(tail -n 1 "$source_file")

    # Check if the last line is different from the current contents of the target file
    if [ "$last_line" != "$(cat "$target_file")" ]
    then
        # Update the target file with the last line of the source file
        echo "$last_line" > "$target_file"
    fi

    # Sleep for 1 second before checking for updates again
    sleep 10
done
