#!/bin/bash


root_folder="baselines"

for subfolder in "$root_folder"/*; do
  if [[ -d "$subfolder" ]]; then

    for subsubfolder in "$subfolder"/*; do
      if [[ -d "$subsubfolder" ]]; then

        for file in "$subsubfolder"/slurm-*.out; do
          if [[ -f "$file" ]] && grep -qi "error" "$file"; then
            echo "Error found in: $file"
          fi
        done

      fi
    done

  fi
done

echo "Finishing Job"
exit 0
