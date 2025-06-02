import glob
import math
import os
import random
import sys
import time
from collections import deque
from threading import Thread
# from typing import List

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major, 
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0]
    )
except IndexError:
    pass

import carla

# ---------------------------------------------------------------------------
# 2. Hyper‑parameters & constants
# ---------------------------------------------------------------------------

# Default staring variables and constant
SHOW_PREVIEW = False # Display actual camera (lock up CPU). True for Debugging
IM_WIDTH = 640      # Default Image setting
IM_HEIGHT = 480     # Default Image setting
SECOND_PER_EPISODE = 10
MEMORY_FRACTION = 0.8
EPISODE = 1_000 # Too few 

# Replay / training cadence
REPLAY_MEMORY_SIZE = 20_000
MIN_REPLAY_MEMORY_SIZE = 5_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 2
UPDATE_TARGET_EVERY = 10

# ε‑greedy
DISCOUNT = 0.99
START_EPSILON = 1
EPSILON_DECAY = 0.9975
MIN_EPSILON = 0.1

# Reward coefficients
SPEED_TARGET = 60
SPEED_WEIGHT  = 0.2
# CTE_WEIGHT = 6.0 # 離中心越遠
# HDG_WEIGHT = 3.0 # 方向偏差

# Checkpoint / logs
MODEL_NAME = "Xception"
MIN_REWARD = -200
AGGREGATE_STATS_EVERY = 10

# Action table  (throttle, brake, steer)AGGREGATE_STATS_EVERY = 10
ACTION_CONTROL = [  # throttle, brake, steer
    (1.0, 0.0,  0.0),   # 1  滿油門直走
    (0.8, 0.0,  0.0),   # 0  中油門直走
    (0.8, 0.0, -0.5),   # 2  微左調整
    (0.8, 0.0,  0.5),   # 3  微右調整
    (0.5, 0.0, -1.0),   # 4  左轉 (較小)
    (0.5, 0.0,  1.0),   # 5  右轉 (較小)
    (0.0, 1.0,  0.0),   # 6  煞車
    (0.0, 0.0,  0.0),   # 7  no‑op
]
ACTION_SPACE_SIZE = len(ACTION_CONTROL)  # 8

# ---------------------------------------------------------------------------
# 3. TensorBoard helper
# ---------------------------------------------------------------------------

class ModifiedTensorBoard(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(log_dir)
        self.step = 0

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        if not logs:      # logs={'loss':..., 'mae':...}
            return
        with self.writer.as_default():
            for k, v in logs.items():
                tf.summary.scalar(k, v, step=self.step)
            self.writer.flush()

    def update_stats(self, **stats):
        with self.writer.as_default():
            for k, v in stats.items():
                tf.summary.scalar(k, v, step=self.step)
            self.writer.flush()

# ---------------------------------------------------------------------------
# 4. CARLA Environment
# ---------------------------------------------------------------------------

class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT

    def __init__(self):
        # World Setting
        self.client = carla.Client("localhost", 3000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.bp_lib = self.world.get_blueprint_library()

        # Vehicle and Sensor bp
        self.model3 = self.bp_lib.filter("model3")[0]
        self.rgb_cam_bp = self.bp_lib.find("sensor.camera.rgb")
        self.rgb_cam_bp.set_attribute("image_size_x", str(IM_WIDTH))
        self.rgb_cam_bp.set_attribute("image_size_y", str(IM_HEIGHT))
        self.rgb_cam_bp.set_attribute("fov", "110")

        # Collision sensor
        self.collision_bp = self.bp_lib.find("sensor.other.collision")

        # Runtime state
        self.actor_list = []
        self.front_camera = None
        self.collision_hist = []
        self.episode_start = None

    def reset(self):
        self._destroy_actors()
        self.collision_hist = []
        self.actor_list = []
        self.front_camera = None
        
        # Vehicle setting
        spawn_start = time.time()
        while True:
            try:
                self.transform = random.choice(self.world.get_map().get_spawn_points())
                self.vehicle = self.world.spawn_actor(self.model3, self.transform)
                break
            except:
                time.sleep(0.01)

            if time.time() > spawn_start + 3:
                raise Exception('Can\'t spawn a car')
            
        self.actor_list.append(self.vehicle)

        # Camera setting
        self.rgb_cam = self.bp_lib.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")

        # Sensor setting
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)

        # Collision Sensor setting
        colsensor = self.bp_lib.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self._on_collision(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.front_camera

    def step(self, action):
        # Action Detecting
        thr, brk, steer = ACTION_CONTROL[action]
        self.vehicle.apply_control(
            carla.VehicleControl(
                throttle=thr, 
                steer=steer * self.STEER_AMT,
                brake=brk
            )
        )

        # Velocity setting
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
    
        # Reward-Penalty Mechanism
        if self.collision_hist:
            reward = MIN_REWARD
            done = True
        else:
            reward = (kmh - SPEED_TARGET) * SPEED_WEIGHT
            done = False

        if time.time() - self.episode_start > SECOND_PER_EPISODE:
            done = True

        return self.front_camera , reward, done, None

    def process_img(self, image):
        i = np.array(image.raw_data)
        # print(i)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3] # height, width, rgb (ignoring alpha value)
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        # return i3/255.0 # Normalize value to [0, 1]
        self.front_camera = i3
    
    def _on_collision(self, event):
        self.collision_hist.append(event)

    def _destroy_actors(self):
        for actor in self.actor_list:
            if hasattr(actor, 'is_listening') and actor.is_listening:
                # actor.stop()
                pass
            if actor.is_alive:
                actor.destroy()

        self.actor_list = []

    def close(self):
        self._destroy_actors()

# ---------------------------------------------------------------------------
# 5. DQN Agent
# ---------------------------------------------------------------------------

class DQNagent():
    def __init__(self):
        
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")

        self.target_update_counter = 0
        self.terminate = False
        self.last_logged_episode = 0

        self.training_initialized = False

    def create_model(self):
        base_model = Xception(weights=None, 
                              include_top=False, 
                              input_shape=(IM_HEIGHT, IM_WIDTH, 3))

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation="relu")(x)

        # 9 Actions
        predictions = Dense(ACTION_SPACE_SIZE, activation="linear")(x)
        model = Model(inputs = base_model.input, outputs = predictions)
        model.compile(
            loss="mse", 
            optimizer=Adam(learning_rate=0.001), 
            metrics=["accuracy"], 
        )

        return model

    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done) # tuple
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

    def act(self, state, epsilon):
        if np.random.rand() < epsilon:
            return random.randrange(ACTION_SPACE_SIZE)
        q_values = self.model.predict(state[np.newaxis, ...] / 255.0, verbose=0)[0]
        return int(np.argmax(q_values))

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        
        current_states  = np.array([transition[0] for transition in minibatch]) / 255.0
        next_states     = np.array([transition[3] for transition in minibatch]) / 255.0
        current_qs_list  = self.model.predict(current_states, batch_size=PREDICTION_BATCH_SIZE, verbose=0)
        next_qs_list = self.target_model.predict(next_states, batch_size=PREDICTION_BATCH_SIZE, verbose=0)

        X = []
        y = []

        for idx, (state, action, reward, new_state, done) in enumerate(minibatch):
            target_q = current_qs_list[idx]
            if done:
                target_q[action] = reward
            else:
                target_q[action] = reward + DISCOUNT * np.max(next_qs_list[idx])
            
            X.append(state)
            y.append(target_q)

        X = np.array(X, dtype=np.float32) / 255.0
        y = np.array(y, dtype=np.float32)

        log = self.tensorboard if self.tensorboard.step > self.last_logged_episode else None
        if log:
            self.last_logged_episode = self.tensorboard.step

        self.model.fit(
            X, y,
            batch_size=TRAINING_BATCH_SIZE,
            verbose=0,
            callbacks=[self.tensorboard] if log else None
        )

        self.target_update_counter += 1

        if self.target_update_counter >= UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]
    
    def train_in_loop(self):
        """Background thread that calls train() continually."""
        X_dummy = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
        y_dummy = np.random.uniform(size=(1, ACTION_SPACE_SIZE)).astype(np.float32)
        self.model.fit(X_dummy, y_dummy, verbose=False, batch_size=1)
        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)

# ---------------------------------------------------------------------------
# 6. Main training loop
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    FPS = 20

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)

    # Memory fraction, used mostly when training multiple agents
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    # GPU Memory Setting (Replacing Session)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print('[WARN] set_memory_growth failed:', e)
    
    # Create models folder
    os.makedirs('models', exist_ok=True)
    
    # Create agent and environment
    agent = DQNagent()
    env = CarEnv()

    # Start training thread and wait for training ti be initialized
    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.01)

    agent.get_qs(np.ones((env.im_height, env.im_width, 3)))
    # For stats
    epsilon = START_EPSILON
    ep_rewards = [-200]

    for episode in tqdm(range(1, EPISODE + 1), ascii=True, unit='episodes'):
        # env.collision_hist = []
        agent.tensorboard.step = episode
        episode_reward = 0
        step = 1
        current_state = env.reset()
        episode_start = time.time()
        done = False
        
        while True:
            # Exploitation
            if np.random.random() > epsilon:
                # Get optimal action from Q table
                action = np.argmax(agent.get_qs(current_state)) # model action (time-optimal)
            # Exploration
            else:
                # Get random action
                action = np.random.randint(0, 3)
                # Add a delay to match FPS (prediction above takes longer)
                time.sleep(1/FPS)

            new_state, reward, done, _ = env.step(action)
            # Transform new continous state to new discrete state and count reward
            episode_reward += reward
            # Every step we update replay memory
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            # agent.train()

            current_state = new_state
            step += 1

            if done:
                break
        
        # End of episode - destroy agents
        for actor in env.actor_list:
            actor.destroy()
        
        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= MIN_REWARD:
                fname = f'models/{MODEL_NAME}__{int(time.time())}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min.model'
                agent.model.save(fname)
                # print(f"- checkpoint saved → {fname}")

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

    # Set termination flag for training thread and wait for it to finish
    agent.terminate = True
    trainer_thread.join()
    
    fname = f'models/{MODEL_NAME}__{int(time.time())}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min.model'
    agent.model.save(fname)
    print(f"- Final Model saved → {fname}")
    
    env.close()
    print("Training Complete.")
