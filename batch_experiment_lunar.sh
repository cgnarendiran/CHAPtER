#!/bin/bash

seedlist=(1 2 3)
num_episodes=1000

echo Starting batch....

for seed in ${seedlist[@]}; do
	echo Running with seed $seed
	./run_experiment_lunar.sh $seed $num_episodes
done
