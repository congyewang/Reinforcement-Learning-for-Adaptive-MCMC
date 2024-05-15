#!/bin/zsh


root_dir=$1
runfile=$2

if [[ -z "$root_dir" || -z "$runfile" ]]; then
    echo "Usage: run_matlab_processing <directory> <runfile>"
fi

cd "$root_dir"

# Calculate the total number of subfolders
total_dirs=$(find . -mindepth 1 -maxdepth 1 -type d | wc -l)
count=0
progress_length=30  # Total length of the progress bar

echo "Starting processing of $root_dir..."

for dir in */; do
    if [[ -f "$dir/$runfile" ]]; then
        ((count++))
        # Calculate the percentage of progress completed
        percent=$((100 * count / total_dirs))
        # Calculate the # to be filled on the progress bar
        filled_length=$((progress_length * count / total_dirs))
        # Generating a progress bar
        bar=$(printf '%*s' "$filled_length" | tr ' ' '#')
        bar=$(printf '%-*s' "$progress_length" "$bar")

        # Use \r to update the same line, \e[K to clear to the end of the line.
        echo -ne "\r[$bar] ${percent}% - Processing ${dir%/}"

        cd "$dir"
        matlab -nodisplay -nosplash -nojvm < $runfile > learning_output.txt 2>&1
        cd ..
    fi
done


# Send Message
url="https://gotify.greenlimes.top"
token="AEqrtS7SCsWFb6J"
curl "$url/message?token=$token" -F "title=${root_dir%/} Finished" -F "message=${root_dir%/} Finished" -F "priority=5" --ssl-no-revoke

echo "Finishing Job"
exit 0
