#!/bin/zsh

matlab -nodisplay -nosplash -nodesktop -r "run('learning.m'); exit;"

mkdir Data
mkdir Data/Policy

python demo_plot.py

mv *.mat Data

matlab -nodisplay -nosplash -nodesktop -r "addpath(genpath('../../src/rlmcmc/')); export_policy_data(); xlims = [-10,10]; n_break1 = 4; n_break2 = 51; polyline = true; plot_demo_nn(xlims, n_break1, n_break2, polyline); exit;"
