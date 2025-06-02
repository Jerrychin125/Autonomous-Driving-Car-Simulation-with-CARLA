# Autonomous-Driving-Car-Simulation-with-CARLA

An end-to-end Deep Q-Learning pipeline that teaches a Tesla Model 3 to drive inside the CARLA simulator.

---

## Abstract
Modern autonomous-driving stacks通常結合多重感測器、地圖與規則型規劃（rule-based planning）。  
本專案嘗試走向另一個極端──**「影像 → 行為」的端到端強化學習**：

* **環境**　CARLA 0.9.15，Town 10，單一 RGB 相機 (640 × 480 @ 20 FPS)  
* **代理**　DQN + Xception backbone（無 ImageNet 預訓練），8 個離散動作 (throttle / brake / steer)  
* **訓練策略**　
  - Replay buffer 20 k，5 k 後開始學習  
  - ε-greedy (ε=1→0.1, γ=0.9975)  
  - Target network 每 10 步同步  
  - Reward = `(v_kmh − 60) × 0.2`，碰撞立即 −200 終止  
* **結果**　約 1 k episodes、4 h (RTX 3060) 後，平均可於 10 s 內保持 58 ~ 62 km/h 且零碰撞  
* **特色**　採用背景執行緒非同步更新，最大化 CARLA server 使用率

---

## Project goals
* **End-to-end learning**：640 × 480 RGB → 8 種離散控制動作  
* **Learned, not scripted**：完全由 reward 信號 (速度 + 碰撞) 引導，沒有 way-points 或 rule-based planner  
* **Asynchronous training**：主執行緒收集資料，背景執行緒持續 `model.fit()`，使 simulator 100 % 忙碌

---

## Key features
* **Replay buffer**：20 000 transitions，累積 5 000 後開始更新  
* **ε-greedy exploration**：ε 由 1.0 緩降至 0.1；γ = 0.9975  
* **Target network**：每 10 次訓練步同步權重，穩定 bootstrap  
* **Speed-based reward**：`r = (v_kmh − 60) × 0.2`，碰撞 -200 立即結束


## Instructions
---

#### Setting Environment 
Build virtual python environment (strictly python3.7),
```py -3.7 -m venv .venv```
```.\.venv\Scripts\activate```
```source .venv/bin/activate```   # Windows
```pip install -r requirements.txt```


#### Running Simulator
```running the simulator .\CarlaUE4.exe -carla-rpc-port=3000 -quality-level=Epic -ResX=1280 -ResY=720 -windowed```

#### Training
Checkpoints are saved to models/ whenever the worst reward over the last 10 episodes ≥ −200.
```py -3.7 carla_dqn_train.py```


### TensorBoard
```tensorboard --logdir logs```

### Playing a Model
```py -3.7 play_model.py --model-path models\your_checkpoint.model```

### Plot
```py -3.7 plot_metrics.py reward.csv loss.csv```
