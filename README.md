# Autonomous-Driving-Car-Simulation-with-CARLA

1. project goals
    - End-to-end learning: From a 640 × 480 RGB frame to 8 discrete control actions
    - Learned, not scripted: No way-points, no rule-based planner — collisions and speed define the reward.
    - Asynchronous training: Data collection runs in the main thread while a background thread updates the network, keeping the simulator busy 100 % of the time. 

2. key features
    - Replay buffer: 20 000 transitions with uniform sampling; starts learning after 5 000 stored steps.
    - ε-greedy exploration: ε decays from 1.0 → 0.1 with γ = 0.9975.
    - Target network: Synchronised every 10 optimisation steps to stabilise bootstrapping.
    - Speed-based reward: r = (v_kmh − 60) × 0.2, plus a −200 penalty for any collision.

3. some instructions

    - environment 
        python -m venv .venv
        source .venv/bin/activate   # Linux / macOS
        pip install -r requirements.txt

    - running the simulator 
        .\CarlaUE4.exe -carla-rpc-port=3000 `
                -quality-level=Epic `
                -ResX=1280 -ResY=720 -windowed

    - training
        py -3.7 carla_dqn_train.py
            - (Checkpoints are saved to models/ whenever the worst reward over the last 10 episodes ≥ −200.)

    - TensorBoard
        tensorboard --logdir logs

    -  Playing a Model
        py -3.7 play_model.py --model-path models\your_checkpoint.model
