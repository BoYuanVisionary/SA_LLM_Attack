#!/bin/bash

# # Loop over cooling rate values
# for cooling_rate in 0.65
# do
#   # Loop over seed values from 0 to 4 for simulated annealing testing
#   for seed in {8..19}
#   do
#     device=$((seed % 4))
#     nohup python test_sa.py --seed "$seed" --batch_size 128 --device "$device" --cooling_rate "$cooling_rate" > "./repeat/sa_output_seed_${seed}.log" 2>&1 &
#     if (( seed % 4 == 3 )); then
#       wait
#     fi
#   done
#   wait
# done

 
# Loop over seed values from 0 to 4 for gradient descent testing
for seed in {0..7}
do
  device=$((seed % 4))
  nohup python test_gd.py --seed "$seed" --batch_size 200 --device "$device" > "./min_values/gd_output_seed_${seed}.log" 2>&1 &
  if (( seed % 4 == 3 )); then
    wait
  fi
done

# for cooling_rate in 0.6
# do
#   for test_length in 20
#   do
#     # Loop over seed values from 0 to 7 for simulated annealing testing
#     for seed in {0..7}
#     do
#       device=$((seed % 4))
#       nohup python test_sa.py --seed "$seed" --batch_size 200 --test_length "$test_length" --device "$device" --cooling_rate "$cooling_rate" > "./min_values/sa_output_seed_${seed}_test_length_${test_length}.log" 2>&1 &
#       if (( seed % 4 == 3 )); then
#         wait
#       fi
#     done
#     wait
#   done
# done
