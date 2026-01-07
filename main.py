import pygame
import sys
import matplotlib.pyplot as plt # 用來畫訓練曲線
import src.config as cfg
from src.env import GolfEnv
from src.agent import QLearningAgent
import numpy as np

def train(episodes=3000):

    # 初始化
    env = GolfEnv()
    agent = QLearningAgent()
    
    # 紀錄數據用來畫圖
    rewards_history = []
    
    print(f"Start Training for {episodes} episodes...")
    print("---------------------------------------------")

    for e in range(episodes):
        # 重置環境
        state = env.reset()
        
        # 決定動作 (is_training T F)
        action = agent.choose_action(state, is_training=True)
        
        # 執行
        next_state, reward, done, _ = env.step(action)
        
        # 學習/更新 q table
        agent.learn(state, action, reward)
        
        # 降低亂試
        agent.decay_epsilon()
        
        # 紀錄
        rewards_history.append(reward)

        # 每 100 回合印出一次進度
        if (e + 1) % 100 == 0:
            avg_reward = sum(rewards_history[-100:]) / 100
            print(f"Episode {e+1}/{episodes} | Epsilon: {agent.epsilon:.2f} | Avg Reward: {avg_reward:.1f}")

    print("---------------------------------------------")
    print("Training Complete!")
    
    # 存檔
    agent.save_model()
    
    # 畫圖表
    # 計算移動平均
    window_size = 10000 
    moving_avg = np.convolve(rewards_history, np.ones(window_size)/window_size, mode='valid')

    plt.figure(figsize=(10, 6))

    # 畫原始數據
    plt.plot(rewards_history, label='Raw Reward', color='lightblue', alpha=0.3)

    # 畫成長曲線
    plt.plot(range(window_size-1, len(rewards_history)), moving_avg, label='Trend (Moving Avg)', color='red', linewidth=2)

    plt.title("Learning Curve: Reward Growth Process")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()  # 顯示圖例
    plt.grid(True) # 加格線比較好讀
    plt.show()
    
    return agent

def demo(agent):

    # 展示模式看 AI 表演
    print("Starting Demo Mode (Press Q to Quit)...")
    
    env = GolfEnv()
    
    running = True
    while running:
        # 處理輸入 (可以按 X 關掉視窗)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False

        # 重置
        state = env.reset()
        
        # is_training=False
        action = agent.choose_action(state, is_training=False)
        
        # 執行與取得路徑
        _, reward, done, path_points = env.step(action)
        
        # 4. 渲染畫面 (這時候才畫圖)
        env.render(path_points)
        
        print(f"Hole: {env.hole_pos}, Reward: {reward:.1f}")
        
        pygame.time.delay(1000) # 1s

    pygame.quit()
    sys.exit()

if __name__ == "__main__":

    # 先訓練
    trained_agent = train(episodes=12000000) # 訓練次數
    
    # 再演示
    demo(trained_agent)