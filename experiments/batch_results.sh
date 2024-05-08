#!/bin/zsh


root_dir="results"

cd "$root_dir"

for dir in */; do
    if [[ -f "$dir/learning.m" ]]; then
        cd "$dir"
        echo "\nStart running $dir\n"
        start=$(date +%s)

        matlab -nodisplay -nosplash -nojvm < learning.m

        end=$(date +%s)
        echo "\nFinish running $dir\n"

        cd ..

        runtime=$((end - start))
        echo "$dir, $runtime" >> results_runtime.txt
    fi
done

echo "\nAll done!\n"
