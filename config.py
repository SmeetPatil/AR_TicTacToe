"""
Configuration file for AR Tic-Tac-Toe
Adjust these settings based on your setup and ball color
"""

import numpy as np

class Config:
    # Grid settings
    GRID_SIZE = 3
    CELL_SIZE = 150  # Size of each cell in pixels
    GRID_OFFSET_X = 100  # X offset from screen edge
    GRID_OFFSET_Y = 100  # Y offset from screen edge
    
    # Ball detection settings (HSV color space)
    # Red ball detection (adjust these values for your specific ball)
    BALL_COLOR_LOWER = np.array([0, 100, 100])   # Lower HSV bound
    BALL_COLOR_UPPER = np.array([10, 255, 255])  # Upper HSV bound
    
    # Alternative color ranges for different colored balls:
    # Blue ball: lower=[100, 100, 100], upper=[130, 255, 255]
    # Green ball: lower=[40, 100, 100], upper=[80, 255, 255]
    # Yellow ball: lower=[20, 100, 100], upper=[30, 255, 255]
    
    # Ball size constraints
    MIN_BALL_RADIUS = 10
    MAX_BALL_RADIUS = 50
    
    # Game timing
    DETECTION_COOLDOWN = 2.0  # Seconds between ball detections
    AI_MOVE_DELAY = 1.0       # Delay before AI makes a move
    
    # Camera settings
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    MIRROR_CAMERA = True  # Flip camera horizontally
    
    # Colors (BGR format for OpenCV)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    YELLOW = (0, 255, 255)
    
    # UI settings
    FONT_SCALE = 0.7
    FONT_THICKNESS = 2
    LINE_THICKNESS = 3