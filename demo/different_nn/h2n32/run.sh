#!/bin/zsh

matlab -nodisplay -nosplash -nodesktop -r "run('learning.m'); exit;"

mkdir Data
mkdir Data/Policy

python demo_plot.py

mv *.mat Data

matlab -nodisplay -nosplash -nodesktop -r "addpath(genpath('../../../src/rlmcmc/')); export_policy_data(); plot_demo_nn(); exit;"
