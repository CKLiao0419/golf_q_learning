import numpy as np

# 視窗定義
DISPLAY_SCREEN_W  = 400
DISPLAY_SCREEN_H  = 600
DISPLAY_FPS       = 60
DISPLAY_TITLE     = "Golf_Q_Learning"

# 顏色定義
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (20,  20,  20)
COLOR_GREEN = (34,  139, 34)
COLOR_GRAY  = (200, 200, 200)
COLOR_RED   = (230, 50,  50)
COLOR_BLUE  = (50,  50,  230)

# 球
BALL_RAD = 7
BALL_POS = (DISPLAY_SCREEN_W // 2, 500) # (x, y)

# 洞
HOLE_RAD     = 13
HOLE_Y_RANGE = (50, 280) # (x, y)

# 物理
PHYS_FRICTION        = 0.96 # 地面摩擦力 (每幀速度 * 0.96)
PHYS_MIN_VELOCITY    = 0.5  # 低於就視為靜止
PHYS_MAX_POWER_SPEED = 25   # 力道

# 人物設定
ACTION_ANGLES = np.linspace(-45, 45, 51)
ACTION_FORCES = np.linspace(0.2, 1.0, 13)
ACTIONS_NUM   = len(ACTION_ANGLES) * len(ACTION_FORCES) # 動作總數

# 空間、網格
GRID_SIZE = 6 # 格子尺寸
GRID_W = DISPLAY_SCREEN_W // GRID_SIZE
GRID_H = DISPLAY_SCREEN_H // GRID_SIZE

# 訓練參數
TRAIN_LEARNING_RATE   = 0.17       # 學得有多快
TRAIN_DISCOUNT_FACTOR = 0.95       # 對未來獎勵的重視程度
TRAIN_EPSILON_START   = 1.0        # 初始探索率
TRAIN_EPSILON_MIN     = 0.01       # 最低探索率 (最後保留 1% 亂試)
TRAIN_EPSILON_DECAY   = 0.9999997  # 每次衰減多少