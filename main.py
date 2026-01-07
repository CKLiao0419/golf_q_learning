import pygame
import sys
import matplotlib.pyplot as plt
import time
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
    
    # 初始化監控視窗
    pygame.init()
    width, height = 500, 250
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Q-Learning Training Monitor")
    
    # 字體設定
    font = pygame.font.SysFont("Arial", 24)
    small_font = pygame.font.SysFont("Arial", 18)
    
    start_time = time.time()
    last_update_time = start_time
    # ------------------------------

    print(f"Start Training for {episodes} episodes...")
    print("---------------------------------------------")

    for e in range(episodes):
        # 1. 處理視窗事件 (避免視窗卡死/無回應)
        pygame.event.pump() 

        # 重置環境
        state = env.reset()
        
        # 決定動作
        action = agent.choose_action(state, is_training=True)
        
        # 執行
        next_state, reward, done, _ = env.step(action)
        
        # 學習
        agent.learn(state, action, reward)
        
        # 降低亂試
        agent.decay_epsilon()
        
        # 紀錄
        rewards_history.append(reward)

        # 每 5 秒更新一次視窗內容
        current_time = time.time()
        if current_time - last_update_time > 5: # 每 5 秒
            
            # 計算平均獎勵 
            recent_avg_reward = sum(rewards_history[-1000:]) / 1000 if len(rewards_history) > 1000 else 0
            progress = (e / episodes) * 100
            elapsed_time = current_time - start_time
            
            # 畫面塗黑
            screen.fill((30, 30, 30)) 
            
            # 準備文字資訊
            texts = [
                f"Training Progress: {progress:.2f}%",
                f"Episode: {e} / {episodes}",
                f"Epsilon (Explore Rate): {agent.epsilon:.4f}",
                f"Recent Avg Reward (1k): {recent_avg_reward:.2f}",
                f"Time Elapsed: {elapsed_time:.0f} sec"
            ]
            
            # 繪製文字
            for i, text_str in enumerate(texts):
                color = (0, 255, 0) if i == 0 else (255, 255, 255) # 第一行綠色，其他白色
                text_surface = font.render(text_str, True, color)
                screen.blit(text_surface, (20, 20 + i * 40))
            
            # 繪製進度條框框
            pygame.draw.rect(screen, (100, 100, 100), (20, 210, width - 40, 20), 2)
            
            # 繪製進度條內容
            pygame.draw.rect(screen, (0, 255, 0), (22, 212, (width - 44) * (e / episodes), 16))

            pygame.display.flip()
            
            # 印出到 Console
            print(f"Update: Ep {e} | Eps: {agent.epsilon:.2f} | Avg: {recent_avg_reward:.2f}")
            
            last_update_time = current_time
        # ---------------------------------------

    print("---------------------------------------------")
    print("Training Complete!")
    
    # 關閉監控以免干擾demo
    pygame.display.quit()
    
    # 存檔
    agent.save_model()
    
    # 畫圖表
    window_size = 10000 
    if len(rewards_history) >= window_size:
        moving_avg = np.convolve(rewards_history, np.ones(window_size)/window_size, mode='valid')
        plt.figure(figsize=(10, 6))

        plt.plot(range(window_size-1, len(rewards_history)), moving_avg, label='Trend', color='red', linewidth=2)
        plt.title("Learning Curve")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return agent

def demo(agent):
    print("Starting Demo Mode (Press Q to Quit)...")
    
    # 初始化 Pygame 
    pygame.init()
    env = GolfEnv() 
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False

        state = env.reset()
        action = agent.choose_action(state, is_training=False)
        _, reward, done, path_points = env.step(action)
        env.render(path_points)
        
        print(f"Hole: {env.hole_pos}, Reward: {reward:.1f}")
        pygame.time.delay(1000)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    
    # 先訓練
    trained_agent = train(episodes=15000000) 
    
    # 後展示
    demo(trained_agent)