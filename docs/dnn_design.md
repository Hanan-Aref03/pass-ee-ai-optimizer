# DNN Design Notes

## Goal

Approximate the optimized PASS configuration from the user layout and QoS so inference is much faster than running `fmincon_pass.m` every time.

## Why The Labels Need Care

The MATLAB optimizer returns a valid configuration, but the PA order within each waveguide is not a mathematically meaningful label ordering. For training stability, the DNN pipeline should treat each waveguide's PA positions as a sorted set before learning.

## Output Structure

- 9 PA x-positions
- 3 waveguide powers

The model should learn both components jointly, but the inference path should still enforce feasibility:

- positions remain inside `[-D/2, D/2]`
- adjacent PAs stay at least `lambda/2` apart
- powers stay nonnegative and do not exceed the BS power budget

## Accuracy Strategy

The starter implementation will use:

- normalized tabular inputs
- a shared multilayer perceptron trunk
- separate heads for positions and powers
- a feasibility projection step at inference time

That combination keeps the model simple while respecting the physical constraints the MATLAB solver already enforces.
