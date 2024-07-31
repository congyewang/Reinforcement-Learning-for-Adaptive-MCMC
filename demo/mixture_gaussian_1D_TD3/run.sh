#!/bin/zsh

# Path of learning.m
FILE="learning.m"

matlab -nodisplay -nosplash -nodesktop -r "run('learning.m'); exit;"

mkdir Data
mkdir Data/Policy

python demo_plot.py

mv *.mat Data

# Regular Expression Matching 10 or 1e1
PATTERN="agent.AgentOptions.ActorOptimizerOptions.GradientThreshold *= *(10|1e1);"

# Check for GradientThreshold in the File
if grep -Eq "$PATTERN" "$FILE"; then
    # Check if the Line is Commented Out
    if grep -Eq "^%.*$PATTERN" "$FILE"; then
        # Line is commented out, Gradient Clipping is Inf
        matlab -nodisplay -nosplash -nodesktop -r "addpath(genpath('../../src/rlmcmc/')); export_policy_data(); xlims = [-10,10]; n_break1 = 13; n_break2 = 60; polyline = true; plot_demo_1D(xlims, n_break1, n_break2, polyline); exit;"
    else
        # Gradient Clipping is 10
        matlab -nodisplay -nosplash -nodesktop -r "addpath(genpath('../../src/rlmcmc/')); export_policy_data(); xlims = [-10,10]; n_break1 = 8; n_break2; polyline = true; plot_demo_1D(xlims, n_break1, n_break2, polyline); exit;"
    fi
else
    # Line does not exist, Gradient Clipping is Inf
    matlab -nodisplay -nosplash -nodesktop -r "addpath(genpath('../../src/rlmcmc/')); export_policy_data(); xlims = [-10,10]; n_break1 = 13; n_break2 = 60; polyline = true; plot_demo_1D(xlims, n_break1, n_break2, polyline); exit;"
fi
