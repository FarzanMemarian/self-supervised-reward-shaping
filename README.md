# Run the code
to run the code, go to the main directory and run some command like the following

python main.py --main-function "train_sparse_rank"  --env-name "HalfCheetah-v2" --run-type "main_opt_demos" --algo ppo --use-gae --lr 3.0e-4 --rew-lr 0.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --ppo-epoch 10 --gamma 0.99 --gae-lambda 0.95 --num-env-steps-tr 81920000 --num-processes 8 --num-steps 2048 --num-mini-batch 4 --log-interval 1 --entropy-coef 0.1 --save-interval 25  --use-proper-time-limits --rew-sign "free" --num-rew-nets 1 --rew-cntr "True" --cntr-coeff 1 --rew-coeff 1  --num-opt-demo 0 --sparse-rew-type "episodic" --shaping  --min-good-trajs 0 --discounted-rew --skip-rew-eval --sparseness 0.3 --skip-rew-eval

# base code

This code is developed based on the following implementation of the PPO algorithm:

    @misc{pytorchrl,
      author = {Kostrikov, Ilya},
      title = {PyTorch Implementations of Reinforcement Learning Algorithms},
      year = {2018},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail}},
    }


    