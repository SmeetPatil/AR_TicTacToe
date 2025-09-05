import cv2
import numpy as np
import sys
import time
from enum import Enum
from collections import deque

class Player(Enum):
    HUMAN = 1
    AI = 2
    NONE = 0

class GameState(Enum):
    PLAYING = 1
    HUMAN_WON = 2
    AI_WON = 3
    DRAW = 4

class ARTicTacToe:
    def __init__(self):
        # Game settings
        self.GRID_SIZE = 3
        self.CELL_SIZE = 200
        self.GRID_OFFSET_X = 50
        self.GRID_OFFSET_Y = 50
        
        # Grid window size
        self.GRID_WINDOW_WIDTH = 800
        self.GRID_WINDOW_HEIGHT = 700
        
        # Colors (BGR format for OpenCV)
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (0, 0, 255)
        self.BLUE = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.YELLOW = (0, 255, 255)
        self.ORANGE = (0, 165, 255)
        
        # Green sword detection parameters
        self.sword_color_lower = np.array([40, 50, 50])
        self.sword_color_upper = np.array([80, 255, 255])
        self.min_sword_area = 500
        self.max_sword_area = 50000
        
        # Stability tracking
        self.stability_frames = 15
        self.position_tolerance = 25
        self.position_history = deque(maxlen=self.stability_frames)
        self.confirmation_counter = 0
        self.required_confirmations = 20
        
        # Game state
        self.board = [[Player.NONE for _ in range(3)] for _ in range(3)]
        self.current_player = Player.HUMAN
        self.game_state = GameState.PLAYING
        self.last_detection_time = 0
        self.detection_cooldown = 1.5
        
        # Camera setup
        self.cap = None
        self.setup_camera()
        
    def setup_camera(self):
        camera_index = None
        try:
            with open('camera_config.txt', 'r') as f:
                camera_index = int(f.read().strip())
            print(f"Using saved camera index: {camera_index}")
        except:
            print("Using default camera 0")
            camera_index = 0
        
        self.cap = cv2.VideoCapture(camera_index)
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                print(f"✓ Camera {camera_index} working! Resolution: {frame.shape[1]}x{frame.shape[0]}")
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                return
            else:
                self.cap.release()
        
        print("Error: Could not open camera")
        sys.exit(1)
    
    def detect_green_sword(self, frame):
        """Detect green sword and return its tip position"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.sword_color_lower, self.sword_color_upper)
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if self.min_sword_area <= area <= self.max_sword_area:
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Get centroid
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Find extreme points
                    topmost = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
                    bottommost = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])
                    leftmost = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
                    rightmost = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])
                    
                    # Calculate distances from centroid
                    dist_top = np.sqrt((topmost[0] - cx)**2 + (topmost[1] - cy)**2)
                    dist_bottom = np.sqrt((bottommost[0] - cx)**2 + (bottommost[1] - cy)**2)
                    dist_left = np.sqrt((leftmost[0] - cx)**2 + (leftmost[1] - cy)**2)
                    dist_right = np.sqrt((rightmost[0] - cx)**2 + (rightmost[1] - cy)**2)
                    
                    # Find farthest point (sword tip)
                    distances = [
                        (dist_top, topmost, "top"),
                        (dist_bottom, bottommost, "bottom"),
                        (dist_left, leftmost, "left"),
                        (dist_right, rightmost, "right")
                    ]
                    
                    distances.sort(key=lambda x: x[0], reverse=True)
                    sword_tip = distances[0][1]
                    tip_direction = distances[0][2]
                    sword_handle = distances[1][1]
                    
                    return {
                        'tip': sword_tip,
                        'handle': sword_handle,
                        'centroid': (cx, cy),
                        'contour': largest_contour,
                        'area': area,
                        'mask': mask,
                        'bbox': (x, y, w, h),
                        'tip_direction': tip_direction,
                        'all_extremes': {
                            'top': topmost,
                            'bottom': bottommost,
                            'left': leftmost,
                            'right': rightmost
                        }
                    }
        
        return None
    
    def is_position_stable(self, position):
        if not position:
            self.position_history.clear()
            return False, None
        
        self.position_history.append(position)
        
        if len(self.position_history) < 8:
            return False, None
        
        positions = list(self.position_history)
        avg_x = sum(pos[0] for pos in positions) / len(positions)
        avg_y = sum(pos[1] for pos in positions) / len(positions)
        avg_pos = (int(avg_x), int(avg_y))
        
        stable = all(
            abs(pos[0] - avg_x) < self.position_tolerance and 
            abs(pos[1] - avg_y) < self.position_tolerance
            for pos in positions[-8:]
        )
        
        return stable, avg_pos if stable else None
    
    def map_camera_to_grid(self, camera_x, camera_y, camera_width, camera_height):
        norm_x = camera_x / camera_width
        norm_y = camera_y / camera_height
        
        grid_x = self.GRID_OFFSET_X + norm_x * (self.GRID_SIZE * self.CELL_SIZE)
        grid_y = self.GRID_OFFSET_Y + norm_y * (self.GRID_SIZE * self.CELL_SIZE)
        
        return int(grid_x), int(grid_y)
    
    def get_grid_cell(self, x, y):
        rel_x = x - self.GRID_OFFSET_X
        rel_y = y - self.GRID_OFFSET_Y
        
        if (0 <= rel_x <= self.GRID_SIZE * self.CELL_SIZE and 
            0 <= rel_y <= self.GRID_SIZE * self.CELL_SIZE):
            
            cell_x = rel_x // self.CELL_SIZE
            cell_y = rel_y // self.CELL_SIZE
            
            return int(cell_x), int(cell_y)
        
        return None
    
    def is_valid_move(self, row, col):
        return (0 <= row < 3 and 0 <= col < 3 and 
                self.board[row][col] == Player.NONE)
    
    def make_move(self, row, col, player):
        if self.is_valid_move(row, col):
            self.board[row][col] = player
            return True
        return False
    
    def check_winner(self):
        # Check rows
        for row in self.board:
            if row[0] == row[1] == row[2] != Player.NONE:
                return row[0]
        
        # Check columns
        for col in range(3):
            if (self.board[0][col] == self.board[1][col] == 
                self.board[2][col] != Player.NONE):
                return self.board[0][col]
        
        # Check diagonals
        if (self.board[0][0] == self.board[1][1] == 
            self.board[2][2] != Player.NONE):
            return self.board[0][0]
        
        if (self.board[0][2] == self.board[1][1] == 
            self.board[2][0] != Player.NONE):
            return self.board[0][2]
        
        # Check for draw
        if all(self.board[i][j] != Player.NONE 
               for i in range(3) for j in range(3)):
            return "DRAW"
        
        return None
    
    def ai_move(self):
        # Try to win
        for i in range(3):
            for j in range(3):
                if self.is_valid_move(i, j):
                    self.board[i][j] = Player.AI
                    if self.check_winner() == Player.AI:
                        return i, j
                    self.board[i][j] = Player.NONE
        
        # Try to block
        for i in range(3):
            for j in range(3):
                if self.is_valid_move(i, j):
                    self.board[i][j] = Player.HUMAN
                    if self.check_winner() == Player.HUMAN:
                        self.board[i][j] = Player.AI
                        return i, j
                    self.board[i][j] = Player.NONE
        
        # Take center
        if self.is_valid_move(1, 1):
            self.board[1][1] = Player.AI
            return 1, 1
        
        # Take corners
        corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        for row, col in corners:
            if self.is_valid_move(row, col):
                self.board[row][col] = Player.AI
                return row, col
        
        # Take any spot
        for i in range(3):
            for j in range(3):
                if self.is_valid_move(i, j):
                    self.board[i][j] = Player.AI
                    return i, j
        
        return None 
   
    def create_grid_window(self):
        grid_frame = np.ones((self.GRID_WINDOW_HEIGHT, self.GRID_WINDOW_WIDTH, 3), dtype=np.uint8) * 255
        
        # Draw grid lines
        for i in range(4):
            x = self.GRID_OFFSET_X + i * self.CELL_SIZE
            cv2.line(grid_frame, (x, self.GRID_OFFSET_Y), 
                    (x, self.GRID_OFFSET_Y + 3 * self.CELL_SIZE), 
                    self.BLACK, 4)
            
            y = self.GRID_OFFSET_Y + i * self.CELL_SIZE
            cv2.line(grid_frame, (self.GRID_OFFSET_X, y), 
                    (self.GRID_OFFSET_X + 3 * self.CELL_SIZE, y), 
                    self.BLACK, 4)
        
        # Draw symbols
        for i in range(3):
            for j in range(3):
                center_x = self.GRID_OFFSET_X + j * self.CELL_SIZE + self.CELL_SIZE // 2
                center_y = self.GRID_OFFSET_Y + i * self.CELL_SIZE + self.CELL_SIZE // 2
                
                if self.board[i][j] == Player.NONE:
                    cv2.putText(grid_frame, f"({i},{j})", 
                               (center_x - 30, center_y + 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
                elif self.board[i][j] == Player.HUMAN:
                    offset = self.CELL_SIZE // 3
                    cv2.line(grid_frame, 
                            (center_x - offset, center_y - offset),
                            (center_x + offset, center_y + offset),
                            self.RED, 8)
                    cv2.line(grid_frame, 
                            (center_x + offset, center_y - offset),
                            (center_x - offset, center_y + offset),
                            self.RED, 8)
                else:
                    radius = self.CELL_SIZE // 3
                    cv2.circle(grid_frame, (center_x, center_y), radius, self.BLUE, 8)
        
        # Game status
        if self.game_state == GameState.PLAYING:
            if self.current_player == Player.HUMAN:
                text = "Your Turn - Point with Green Sword!"
                color = self.RED
            else:
                text = "AI is Thinking..."
                color = self.BLUE
        elif self.game_state == GameState.HUMAN_WON:
            text = "You Won!"
            color = self.GREEN
        elif self.game_state == GameState.AI_WON:
            text = "AI Won!"
            color = self.RED
        else:
            text = "It's a Draw!"
            color = self.BLACK
        
        cv2.putText(grid_frame, text, (50, 650), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, color, 3)
        
        # Progress bar
        if self.confirmation_counter > 0:
            progress = self.confirmation_counter / self.required_confirmations
            bar_width = 300
            bar_height = 20
            bar_x = (self.GRID_WINDOW_WIDTH - bar_width) // 2
            bar_y = 680
            
            cv2.rectangle(grid_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), self.BLACK, 2)
            fill_width = int(bar_width * progress)
            cv2.rectangle(grid_frame, (bar_x + 2, bar_y + 2), 
                         (bar_x + fill_width - 2, bar_y + bar_height - 2), self.GREEN, -1)
            
            cv2.putText(grid_frame, "Hold sword steady to confirm...", (bar_x, bar_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.BLACK, 2)
        
        return grid_frame
    
    def reset_game(self):
        self.board = [[Player.NONE for _ in range(3)] for _ in range(3)]
        self.current_player = Player.HUMAN
        self.game_state = GameState.PLAYING
        self.last_detection_time = 0
        self.position_history.clear()
        self.confirmation_counter = 0
    
    def run(self):
        print("AR Tic-Tac-Toe with Green Sword started!")
        print("Two windows will open:")
        print("1. 'Game Grid' - Cast this window to your TV")
        print("2. 'Camera Feed' - Shows sword detection (NOT flipped)")
        print("\nHow to play:")
        print("- Point your GREEN SWORD at a cell")
        print("- Hold position steady for confirmation")
        print("- Press 'q' to quit, 'r' to restart")
        
        while True:
            ret, camera_frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            # Don't flip camera frame - keep it natural
            camera_height, camera_width = camera_frame.shape[:2]
            
            grid_frame = self.create_grid_window()
            
            # Handle sword detection for human player
            if (self.game_state == GameState.PLAYING and 
                self.current_player == Player.HUMAN):
                
                current_time = time.time()
                sword_detection = self.detect_green_sword(camera_frame)
                
                if sword_detection:
                    # Draw sword detection
                    cv2.drawContours(camera_frame, [sword_detection['contour']], -1, self.GREEN, 2)
                    
                    x, y, w, h = sword_detection['bbox']
                    cv2.rectangle(camera_frame, (x, y), (x + w, y + h), self.YELLOW, 2)
                    
                    # Draw sword tip (large red dot)
                    cv2.circle(camera_frame, sword_detection['tip'], 10, self.RED, -1)
                    cv2.circle(camera_frame, sword_detection['handle'], 6, self.BLUE, -1)
                    cv2.circle(camera_frame, sword_detection['centroid'], 4, self.WHITE, -1)
                    
                    # Draw line from handle to tip
                    cv2.line(camera_frame, sword_detection['handle'], sword_detection['tip'], self.ORANGE, 3)
                    
                    # Draw all extreme points for debugging
                    extremes = sword_detection['all_extremes']
                    cv2.circle(camera_frame, extremes['top'], 3, (255, 255, 0), -1)
                    cv2.circle(camera_frame, extremes['bottom'], 3, (255, 0, 255), -1)
                    cv2.circle(camera_frame, extremes['left'], 3, (0, 255, 255), -1)
                    cv2.circle(camera_frame, extremes['right'], 3, (128, 0, 128), -1)
                    
                    # Label the tip
                    tip_pos = sword_detection['tip']
                    cv2.putText(camera_frame, f"TIP ({sword_detection['tip_direction']})", 
                               (tip_pos[0] + 15, tip_pos[1] - 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.RED, 2)
                    
                    cv2.putText(camera_frame, f"Area: {int(sword_detection['area'])}", 
                               (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.WHITE, 1)
                    
                    # Check stability
                    is_stable, stable_pos = self.is_position_stable(sword_detection['tip'])
                    
                    if is_stable and stable_pos:
                        grid_x, grid_y = self.map_camera_to_grid(stable_pos[0], stable_pos[1], 
                                                               camera_width, camera_height)
                        
                        cell = self.get_grid_cell(grid_x, grid_y)
                        
                        if cell:
                            col, row = cell
                            if self.is_valid_move(row, col):
                                cv2.circle(camera_frame, stable_pos, 25, self.YELLOW, 3)
                                cv2.putText(camera_frame, f"Cell ({row},{col})", 
                                           (stable_pos[0] + 30, stable_pos[1]), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.YELLOW, 2)
                                
                                self.confirmation_counter += 1
                                
                                if self.confirmation_counter >= self.required_confirmations:
                                    if current_time - self.last_detection_time > self.detection_cooldown:
                                        if self.make_move(row, col, Player.HUMAN):
                                            print(f"Human played at ({row}, {col})")
                                            self.last_detection_time = current_time
                                            self.confirmation_counter = 0
                                            self.position_history.clear()
                                            
                                            winner = self.check_winner()
                                            if winner == Player.HUMAN:
                                                self.game_state = GameState.HUMAN_WON
                                            elif winner == Player.AI:
                                                self.game_state = GameState.AI_WON
                                            elif winner == "DRAW":
                                                self.game_state = GameState.DRAW
                                            else:
                                                self.current_player = Player.AI
                            else:
                                self.confirmation_counter = 0
                                cv2.putText(camera_frame, "Cell occupied!", 
                                           (stable_pos[0] + 30, stable_pos[1] + 30), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.RED, 2)
                        else:
                            self.confirmation_counter = 0
                            cv2.putText(camera_frame, "Point at grid!", 
                                       (stable_pos[0] + 30, stable_pos[1] + 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.ORANGE, 2)
                    else:
                        self.confirmation_counter = 0
                        cv2.putText(camera_frame, "Hold steady!", 
                                   (sword_detection['tip'][0] + 30, 
                                    sword_detection['tip'][1] + 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.ORANGE, 2)
                else:
                    self.confirmation_counter = 0
                    cv2.putText(camera_frame, "Show green sword!", 
                               (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.RED, 2)
            
            # Handle AI move
            elif (self.game_state == GameState.PLAYING and 
                  self.current_player == Player.AI):
                
                time.sleep(1)
                ai_move = self.ai_move()
                if ai_move:
                    row, col = ai_move
                    print(f"AI played at ({row}, {col})")
                    
                    winner = self.check_winner()
                    if winner == Player.HUMAN:
                        self.game_state = GameState.HUMAN_WON
                    elif winner == Player.AI:
                        self.game_state = GameState.AI_WON
                    elif winner == "DRAW":
                        self.game_state = GameState.DRAW
                    else:
                        self.current_player = Player.HUMAN
            
            # Instructions
            instructions = [
                "Point GREEN SWORD at a cell",
                "Hold steady for confirmation",
                "Press 'q' to quit, 'r' to restart"
            ]
            
            for i, instruction in enumerate(instructions):
                cv2.putText(camera_frame, instruction, (10, 30 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.WHITE, 1)
            
            # Show progress
            if self.confirmation_counter > 0:
                progress_text = f"Confirming: {int((self.confirmation_counter/self.required_confirmations)*100)}%"
                cv2.putText(camera_frame, progress_text, (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.YELLOW, 2)
            
            # Show mask in corner
            if 'sword_detection' in locals() and sword_detection:
                mask_small = cv2.resize(sword_detection['mask'], (160, 120))
                mask_colored = cv2.applyColorMap(mask_small, cv2.COLORMAP_JET)
                camera_frame[10:130, camera_width-170:camera_width-10] = mask_colored
            
            cv2.imshow('Game Grid (Cast this to TV)', grid_frame)
            cv2.imshow('Camera Feed - Sword Detection', camera_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.reset_game()
                print("Game reset!")
        
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    try:
        game = ARTicTacToe()
        game.run()
    except KeyboardInterrupt:
        print("\nGame interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()