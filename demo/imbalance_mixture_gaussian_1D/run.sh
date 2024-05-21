#!/bin/zsh

matlab -nodisplay -nosplash -nodesktop -r "run('learning.m'); exit;"

mkdir Data
mkdir Data/Policy

python demo_plot.py

mv *.mat Data

matlab -nodisplay -nosplash -nodesktop -r "addpath(genpath('../../src/rlmcmc/')); export_policy_data(); xlims = [-10,10]; n_break1 = 3; n_break2 = 52; polyline = false; plot_demo_1D(xlims, n_break1, n_break2, polyline); exit;"
