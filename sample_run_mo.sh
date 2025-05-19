#!/bin/bash

echo "Installing required dependencies..."

echo "Running deblurring task with Measurement Optimization..."

python sample_condition.py \
    --model_config configs/model_config.yaml \
    --diffusion_config configs/diffusion_config.yaml \
    --task_config configs/gaussian_deblur_mo_config.yaml \
    --use_mo True \
    --N_sgld_steps 50 \
    --sgld_lr 5e-5 \
    --ddim_eta 0.0 \
    --save_dir ./results_mo \
    --gpu 0

echo "Completed!" 