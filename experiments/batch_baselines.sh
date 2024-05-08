#!/bin/zsh


root_dir="baselines"

cd "$root_dir"

for dir in */; do
    if [[ -f "$dir/baseline.m" ]]; then
        cd "$dir"
        echo "\nStart running $dir\n"
        start=$(date +%s)

        matlab -nodisplay -nosplash -nojvm < baseline.m

        end=$(date +%s)
        echo "\nFinish running $dir\n"

        cd ..

        runtime=$((end - start))
        echo "$dir, $runtime" >> baselines_runtime.txt
    fi
done

echo "\nAll done!\n"
