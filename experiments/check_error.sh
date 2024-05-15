#!/bin/zsh


root_folder="results"

for subfolder in $root_folder/*; do
  if [[ -d $subfolder ]]; then

    for subsubfolder in $subfolder/*; do
      if [[ -d $subsubfolder ]]; then

        file="$subsubfolder/learning_output.txt"

        if [[ -f $file ]] && grep -qi "error" "$file"; then
          echo "Error found in: $file"
        fi

      fi
    done

  fi
done


echo "Finishing Job"
exit 0
