#!/bin/zsh

target_directory="results"

for dir in "${target_directory}"/*/; do
  if [[ -f "${dir}/learning_output.txt" ]]; then
    grep -qi 'error' "${dir}/learning_output.txt"
    if [[ $? -eq 0 ]]; then
      echo "Error found in directory: ${dir}"
    fi
  fi
done

