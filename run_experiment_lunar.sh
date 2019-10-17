#!/bin/bash

seed=$1
num_episodes=$2
memorysize=50000
lr_per=0.000025

echo Running LunarLander with seed $seed
echo "################################"
python main.py --env-name="LunarLander-v2" --alg="dqn" --model-name="LL_vanilla_RER_$seed" --num-episodes=$num_episodes --replay-batch=1 --seed $seed --memory-size $memorysize
python main.py --env-name="LunarLander-v2" --alg="dqn" --model-name="LL_vanilla_PER_$seed" --num-episodes=$num_episodes --priority-replay --seed $seed --memory-size $memorysize --lr $lr_per
python main.py --env-name="LunarLander-v2" --alg="dqn" --model-name="LL_vanilla_HER_$seed" --num-episodes=$num_episodes --hindsight-replay --seed $seed --memory-size $memorysize
python main.py --env-name="lunarsparse-v0" --alg="dqn" --model-name="LL_sparse_RER_$seed" --num-episodes=$num_episodes --replay-batch=1 --seed $seed --memory-size $memorysize
python main.py --env-name="lunarsparse-v0" --alg="dqn" --model-name="LL_sparse_PER_$seed" --num-episodes=$num_episodes --priority-replay --seed $seed --memory-size $memorysize --lr $lr_per
python main.py --env-name="lunarsparse-v0" --alg="dqn" --model-name="LL_sparse_HER_$seed" --num-episodes=$num_episodes --hindsight-replay --seed $seed --memory-size $memorysize
