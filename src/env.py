import pygame
import numpy as np
import math
import src.config as cfg  # 參數設定檔

class GolfEnv:
    def __init__(self):

        # 初始化
        pygame.init()
        self.screen = pygame.display.set_mode((cfg.DISPLAY_SCREEN_W, cfg.DISPLAY_SCREEN_H))
        pygame.display.set_caption(cfg.DISPLAY_TITLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)

        # 遊戲狀態變數
        self.ball_pos = list(cfg.BALL_POS) # [x, y]
        self.hole_pos = [0, 0]             # [x, y]
        self.velocity = [0, 0]             # [vx, vy]

    def reset(self):

        self.ball_pos = list(cfg.BALL_POS)
        self.velocity = [0, 0]

        # 隨機生成洞口
        hole_x = np.random.randint(50, cfg.DISPLAY_SCREEN_W - 50)            # 不要太靠牆
        hole_y = np.random.randint(cfg.HOLE_Y_RANGE[0], cfg.HOLE_Y_RANGE[1])
        self.hole_pos = [hole_x, hole_y]

        return self._get_state()
    
    def step(self, action_idx):

        # 拆解
        n_angles = len(cfg.ACTION_ANGLES)
        
        angle_idx = action_idx % n_angles
        force_idx = action_idx // n_angles

        angle_deg = cfg.ACTION_ANGLES[angle_idx]
        force_val = cfg.ACTION_FORCES[force_idx]

        # 計算初速度
        rad = math.radians(angle_deg)                 # 角度轉弧度

        speed = force_val * cfg.PHYS_MAX_POWER_SPEED
        self.velocity[0] = speed * math.sin(rad)      # 水平分量
        self.velocity[1] = -speed * math.cos(rad)     # 垂直分量 (負號是往上)

        # 物理模擬
        path_points = [] # 紀錄路徑用來畫圖
        running = True
        
        while running:
            # 移動球
            self.ball_pos[0] += self.velocity[0]
            self.ball_pos[1] += self.velocity[1]
            path_points.append(tuple(self.ball_pos))

            # 摩擦力減速
            self.velocity[0] *= cfg.PHYS_FRICTION
            self.velocity[1] *= cfg.PHYS_FRICTION

            # 邊界檢查
            # 左
            if self.ball_pos[0] <= cfg.BALL_RAD:
                self.ball_pos[0] = cfg.BALL_RAD
                self.velocity[0] *= -1 # 反彈
            # 右
            elif self.ball_pos[0] >= cfg.DISPLAY_SCREEN_W - cfg.BALL_RAD:
                self.ball_pos[0] = cfg.DISPLAY_SCREEN_W - cfg.BALL_RAD
                self.velocity[0] *= -1
            # 上
            if self.ball_pos[1] <= cfg.BALL_RAD:
                self.ball_pos[1] = cfg.BALL_RAD
                self.velocity[1] *= -1
            # 下
            elif self.ball_pos[1] >= cfg.DISPLAY_SCREEN_H - cfg.BALL_RAD:
                self.ball_pos[1] = cfg.DISPLAY_SCREEN_H - cfg.BALL_RAD
                self.velocity[1] *= -1

            # 速度太慢就停
            current_speed = math.hypot(self.velocity[0], self.velocity[1])
            if current_speed < cfg.PHYS_MIN_VELOCITY:
                running = False
        
        # 獎勵
        dist = math.hypot(self.ball_pos[0] - self.hole_pos[0], self.ball_pos[1] - self.hole_pos[1]) # 球et洞口距離
        reward = max(0, 100 - (dist / 3)) # 基礎獎勵
        # 進洞獎勵 (如果距離小於洞半徑)
        is_in_hole = dist < cfg.HOLE_RAD
        if is_in_hole:
            reward += 1000 # 超級大獎勵
            print(f"Goal! Dist: {dist:.2f}")

        # just for 格式統一
        next_state = self._get_state()
        done = True 
        
        return next_state, reward, done, path_points
    
    
    def _get_state(self):

        # 回傳洞在哪個格子
        gx = int(self.hole_pos[0] // cfg.GRID_SIZE)
        gy = int(self.hole_pos[1] // cfg.GRID_SIZE)
        
        # 確保不超出邊界
        gx = min(max(0, gx), cfg.GRID_W - 1)
        gy = min(max(0, gy), cfg.GRID_H - 1)

        # 二維陣列攤平成索引
        state_id = gy * cfg.GRID_W + gx
        return state_id

    def render(self, path_points=None):

        self.screen.fill(cfg.COLOR_GREEN)

        # 畫區域分隔線
        pygame.draw.line(self.screen, cfg.COLOR_BLACK, (0, 300), (cfg.DISPLAY_SCREEN_W, 300), 2)
        pygame.draw.line(self.screen, cfg.COLOR_BLACK, (0, 400), (cfg.DISPLAY_SCREEN_W, 400), 2)

        # 畫洞
        pygame.draw.circle(self.screen, cfg.COLOR_BLUE, (int(self.hole_pos[0]), int(self.hole_pos[1])), cfg.HOLE_RAD)

        # 畫球
        pygame.draw.circle(self.screen, cfg.COLOR_GRAY, (int(cfg.BALL_POS[0]), int(cfg.BALL_POS[1])), cfg.BALL_RAD)
        
        # 畫路徑
        if path_points and len(path_points) > 1:
            pygame.draw.lines(self.screen, cfg.COLOR_WHITE, False, path_points, 2)
            # 畫球的最終位置
            last_pos = path_points[-1]
            pygame.draw.circle(self.screen, (200, 100, 100), (int(last_pos[0]), int(last_pos[1])), cfg.BALL_RAD)

        pygame.display.flip()