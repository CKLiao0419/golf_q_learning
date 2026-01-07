import numpy as np
import pickle # 用來存檔et讀檔
import os
import src.config as cfg

class QLearningAgent:
    def __init__(self):
        # 網格總數
        self.n_states = cfg.GRID_W * cfg.GRID_H
        
        # 動作組合總數
        self.n_actions = cfg.ACTIONS_NUM

        # 初始化 q-table，全0
        self.q_table = np.zeros((self.n_states, self.n_actions))

        # 載入訓練參數
        self.lr = cfg.TRAIN_LEARNING_RATE
        self.gamma = cfg.TRAIN_DISCOUNT_FACTOR
        self.epsilon = cfg.TRAIN_EPSILON_START
        self.epsilon_min = cfg.TRAIN_EPSILON_MIN
        self.epsilon_decay = cfg.TRAIN_EPSILON_DECAY

    def choose_action(self, state_id, is_training=True):

        # 在訓練＋隨機數小於 epsilon ---> 隨機亂試
        if is_training and np.random.uniform(0, 1) < self.epsilon:
            # 隨機動
            return np.random.randint(0, self.n_actions)
        else:
            # 選 Q 表中該狀態下分數最高的動作
            return np.argmax(self.q_table[state_id])

    def learn(self, state, action, reward):

        # Q-Learning 核心公式更新：
        # Q(S, A) = Q(S, A) + lr * [R - Q(S, A)]

        # 舊的認知
        old_value = self.q_table[state, action]
        
        # 更新公式 (因為是一次性賽局，Target 就是 Reward)
        target = reward 
        
        # 更新 Q 值
        new_value = old_value + self.lr * (target - old_value)
        self.q_table[state, action] = new_value

    def decay_epsilon(self):
        # 隨著時間越來越少亂試，越來越相信經驗
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, path="models/q_table.pkl"):
        # 存檔
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"Model saved to {path}")

    def load_model(self, path="models/q_table.pkl"):
        # 讀檔
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.q_table = pickle.load(f)
            print(f"Model loaded from {path}")
            self.epsilon = self.epsilon_min 
        else:
            print("No model found, starting from scratch.")