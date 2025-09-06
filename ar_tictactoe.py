import cv2
import numpy as np
import sys
import time
from enum import Enum
from collections import deque

class Player(Enum):
    PLAYER1 = 1
    PLAYER2 = 2
    NONE = 0

class GameState(Enum):
    PLAYING = 1
    ROUND_PLAYER1_WON = 2
    ROUND_PLAYER2_WON = 3
    ROUND_DRAW = 4
    ROUND_BREAK = 5
    MATCH_PLAYER1_WON = 6
    MATCH_PLAYER2_WON = 7

class ARTicTacToe:
    def __init__(self):
        # Game settings
        self.GRID_SIZE = 3
        self.CELL_SIZE = 200
        self.GRID_OFFSET_X = 50
        self.GRID_OFFSET_Y = 20  # Shifted higher on canvas
        
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
        self.CYAN = (255, 255, 0)  # Cyan color for text
        self.ORANGE = (0, 165, 255)
        self.PURPLE = (255, 0, 255)
        
        # Green sword detection parameters
        self.sword_color_lower = np.array([40, 50, 50])
        self.sword_color_upper = np.array([80, 255, 255])
        self.min_sword_area = 500
        self.max_sword_area = 50000
        
        # Load calibrated ball detection parameters
        self.load_ball_calibration()
        
        # Ball detection state tracking - optimized for faster response
        self.ball_detection_history = deque(maxlen=3)  # Reduced history for faster response
        self.last_ball_detection_time = 0
        self.ball_detection_cooldown = 0.5  # Reduced cooldown for faster response
        
        # Motion-based detection optimization
        self.previous_frame = None
        self.motion_threshold = 30  # Minimum motion to trigger detection
        self.ball_velocity_history = deque(maxlen=5)
        self.proximity_detection_enabled = True
        
        # Input method selection
        self.input_method = None  # 'sword' or 'ball'
        self.input_method_selected = False
        
        # Scoreboard window size
        self.SCOREBOARD_WIDTH = 400
        self.SCOREBOARD_HEIGHT = 300
        
        # Stability tracking
        self.stability_frames = 15
        self.position_tolerance = 25
        self.position_history = deque(maxlen=self.stability_frames)
        self.confirmation_counter = 0
        self.required_confirmations = 20
        
        # Game state
        self.board = [[Player.NONE for _ in range(3)] for _ in range(3)]
        self.current_player = Player.PLAYER1
        self.game_state = GameState.PLAYING
        self.last_detection_time = 0
        self.detection_cooldown = 1.5
        
        # Match system (Best of 3)
        self.player1_score = 0
        self.player2_score = 0
        self.current_round = 1
        self.max_rounds = 3
        self.round_break_duration = 15  # seconds
        self.round_break_start_time = 0
        
        # Camera setup
        self.cap = None
        self.setup_camera()
    
    def load_ball_calibration(self):
        """Load calibrated ball detection parameters"""
        try:
            import json
            with open('ball_calibration.json', 'r') as f:
                calibration = json.load(f)
            
            # Load calibrated parameters
            self.yellow_lower = np.array(calibration['yellow_lower'])
            self.yellow_upper = np.array(calibration['yellow_upper'])
            self.min_ball_area = calibration['min_area']
            self.max_ball_area = calibration['max_area']
            self.ball_circularity_threshold = calibration['min_circularity']
            
            print("[OK] Loaded calibrated ball detection parameters")
            print(f"  HSV range: {calibration['yellow_lower']} - {calibration['yellow_upper']}")
            print(f"  Area range: {calibration['min_area']} - {calibration['max_area']}")
            print(f"  Min circularity: {calibration['min_circularity']:.2f}")
            
        except FileNotFoundError:
            # Use more permissive default parameters for better detection
            self.yellow_lower = np.array([10, 80, 80])   # Wider yellow range
            self.yellow_upper = np.array([40, 255, 255])
            self.min_ball_area = 200      # Lower minimum area
            self.max_ball_area = 10000    # Higher maximum area
            self.ball_circularity_threshold = 0.3  # Lower circularity threshold
            
            print("[WARNING] No calibration file found, using permissive default parameters")
            print(f"  HSV range: [10,80,80] - [40,255,255]")
            print(f"  Area range: 200 - 10000")
            print(f"  Min circularity: 0.3")
            print("  Run 'python calibrate_ball.py' to fine-tune detection")
        except Exception as e:
            print(f"[WARNING] Error loading calibration: {e}")
            # Use permissive default parameters
            self.yellow_lower = np.array([10, 80, 80])
            self.yellow_upper = np.array([40, 255, 255])
            self.min_ball_area = 200
            self.max_ball_area = 10000
            self.ball_circularity_threshold = 0.3
        
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
                print(f"[OK] Camera {camera_index} working! Resolution: {frame.shape[1]}x{frame.shape[0]}")
                
                # Optimize camera settings for performance
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)  # Higher FPS for better responsiveness
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
                
                # Try to set additional performance properties
                try:
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                except:
                    pass  # Not all cameras support this
                
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
    
    def detect_tennis_ball(self, frame):
        """Simplified yellow smiley ball detection - always active for debugging"""
        current_time = time.time()
        
        # TEMPORARILY DISABLE motion detection for debugging
        # Store current frame for next iteration
        self.previous_frame = frame.copy()
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create yellow mask with current parameters
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find best ball candidate
        best_ball = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # More lenient area check for debugging
            if area < 200:  # Very low minimum
                continue
                
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # More lenient circularity check
            if circularity < 0.3:  # Very low threshold
                continue
            
            # Get center and radius
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            
            # Score based on area and circularity
            # Prefer larger, more circular objects
            area_score = min(area / 2000, 1.0)  # Normalize area
            circ_score = circularity
            score = (area_score * 0.4) + (circ_score * 0.6)
            
            if score > best_score:
                best_score = score
                
                # Calculate velocity for proximity detection
                if len(self.ball_velocity_history) > 0:
                    last_pos = self.ball_velocity_history[-1]['center']
                    velocity = np.sqrt((center[0] - last_pos[0])**2 + (center[1] - last_pos[1])**2)
                else:
                    velocity = 0
                
                best_ball = {
                    'center': center,
                    'radius': radius,
                    'contour': contour,
                    'area': area,
                    'circularity': circularity,
                    'mask': yellow_mask,
                    'bbox': cv2.boundingRect(contour),
                    'velocity': velocity,
                    'timestamp': current_time,
                    'score': score
                }
        
        if best_ball:
            # Store velocity history
            self.ball_velocity_history.append({
                'center': best_ball['center'],
                'time': current_time,
                'velocity': best_ball['velocity']
            })
            
            # Add to detection history
            self.ball_detection_history.append(best_ball)
            
            return best_ball
        
        return None
    
    def debug_show_all_circles(self, frame):
        """Debug method to show all detected yellow objects with detailed info"""
        # Convert to HSV for yellow detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create yellow mask
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Show current HSV parameters
        cv2.putText(frame, f"HSV: [{self.yellow_lower[0]},{self.yellow_lower[1]},{self.yellow_lower[2]}]", 
                   (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.WHITE, 1)
        cv2.putText(frame, f"     [{self.yellow_upper[0]},{self.yellow_upper[1]},{self.yellow_upper[2]}]", 
                   (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.WHITE, 1)
        cv2.putText(frame, f"Area: {self.min_ball_area}-{self.max_ball_area}", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.WHITE, 1)
        
        valid_detections = 0
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 50:  # Show even small contours for debugging
                # Get bounding circle
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                # Check if this would be a valid detection
                is_valid = (area >= 200 and circularity >= 0.3)
                
                if is_valid:
                    valid_detections += 1
                    color = self.GREEN  # Green for valid
                    thickness = 3
                else:
                    color = self.RED    # Red for invalid
                    thickness = 1
                
                # Draw contour and circle
                cv2.drawContours(frame, [contour], -1, color, thickness)
                cv2.circle(frame, center, radius, color, thickness)
                cv2.circle(frame, center, 3, color, -1)
                
                # Show detailed info
                cv2.putText(frame, f"#{i}", (center[0] - 10, center[1] - radius - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                cv2.putText(frame, f"A:{int(area)}", (center[0] + radius + 5, center[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                cv2.putText(frame, f"C:{circularity:.2f}", (center[0] + radius + 5, center[1] + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                cv2.putText(frame, f"R:{radius}", (center[0] + radius + 5, center[1] + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Show summary
        cv2.putText(frame, f"Total: {len(contours)} | Valid: {valid_detections}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.CYAN, 2)
    
    def is_ball_completely_in_cell(self, ball_center, ball_radius, camera_width, camera_height):
        """Check if the ball is completely inside a grid cell with strict tolerance"""
        # Map ball center to grid coordinates
        grid_x, grid_y = self.map_camera_to_grid(ball_center[0], ball_center[1], 
                                                camera_width, camera_height)
        
        # Get the cell
        cell = self.get_grid_cell(grid_x, grid_y)
        if not cell:
            return False, None
        
        col, row = cell
        
        # Calculate cell boundaries in grid coordinates
        cell_left = self.GRID_OFFSET_X + col * self.CELL_SIZE
        cell_right = cell_left + self.CELL_SIZE
        cell_top = self.GRID_OFFSET_Y + row * self.CELL_SIZE
        cell_bottom = cell_top + self.CELL_SIZE
        
        # Map ball radius to grid scale
        scale_x = (self.GRID_SIZE * self.CELL_SIZE) / camera_width
        scale_y = (self.GRID_SIZE * self.CELL_SIZE) / camera_height
        grid_radius = ball_radius * max(scale_x, scale_y)
        
        # Stricter tolerance for yellow smiley ball - must be well inside
        tolerance = 25  # Larger tolerance for yellow smiley ball
        
        # Check if ball (including radius + tolerance) is completely within cell
        ball_left = grid_x - grid_radius - tolerance
        ball_right = grid_x + grid_radius + tolerance
        ball_top = grid_y - grid_radius - tolerance
        ball_bottom = grid_y + grid_radius + tolerance
        
        is_completely_inside = (ball_left >= cell_left and 
                               ball_right <= cell_right and 
                               ball_top >= cell_top and 
                               ball_bottom <= cell_bottom)
        
        return is_completely_inside, (row, col) if is_completely_inside else None
    
    def process_ball_detection_instant(self, ball_detection, camera_width, camera_height):
        """Process ball detection with proximity-based instant response"""
        current_time = time.time()
        
        if not ball_detection:
            return False
        
        # Check if ball is near or in a cell
        is_inside, cell_pos = self.is_ball_near_cell(
            ball_detection['center'], ball_detection['radius'], 
            camera_width, camera_height)
        
        if is_inside and cell_pos:
            row, col = cell_pos
            if self.is_valid_move(row, col):
                # Enhanced detection logic: check if ball is slowing down (landing)
                is_landing = self.is_ball_landing(ball_detection)
                
                # Instant move when ball is detected in cell and slowing down
                if (current_time - self.last_ball_detection_time > self.ball_detection_cooldown and
                    (is_landing or ball_detection['velocity'] < 20)):  # Low velocity threshold
                    
                    if self.make_move(row, col, self.current_player):
                        player_name = "Player 1" if self.current_player == Player.PLAYER1 else "Player 2"
                        print(f"{player_name} played at ({row}, {col}) with YELLOW SMILEY BALL")
                        self.last_ball_detection_time = current_time
                        
                        winner = self.check_winner()
                        if winner == Player.PLAYER1 or winner == Player.PLAYER2 or winner == "DRAW":
                            self.handle_round_end(winner)
                        else:
                            # Switch to other player
                            self.current_player = Player.PLAYER2 if self.current_player == Player.PLAYER1 else Player.PLAYER1
                        return True
        return False
    
    def is_ball_landing(self, ball_detection):
        """Detect if ball is landing (velocity decreasing)"""
        if len(self.ball_velocity_history) < 3:
            return False
        
        # Check if velocity is decreasing over recent frames
        recent_velocities = [h['velocity'] for h in list(self.ball_velocity_history)[-3:]]
        
        # Ball is landing if velocity is consistently decreasing
        is_decreasing = all(recent_velocities[i] >= recent_velocities[i+1] 
                           for i in range(len(recent_velocities)-1))
        
        # Or if current velocity is very low
        current_velocity = ball_detection.get('velocity', 0)
        is_slow = current_velocity < 15
        
        return is_decreasing or is_slow
    
    def is_ball_near_cell(self, ball_center, ball_radius, camera_width, camera_height):
        """Check if ball is near or in a grid cell with relaxed tolerance for faster detection"""
        # Map ball center to grid coordinates
        grid_x, grid_y = self.map_camera_to_grid(ball_center[0], ball_center[1], 
                                                camera_width, camera_height)
        
        # Get the cell
        cell = self.get_grid_cell(grid_x, grid_y)
        if not cell:
            return False, None
        
        col, row = cell
        
        # Calculate cell boundaries in grid coordinates
        cell_left = self.GRID_OFFSET_X + col * self.CELL_SIZE
        cell_right = cell_left + self.CELL_SIZE
        cell_top = self.GRID_OFFSET_Y + row * self.CELL_SIZE
        cell_bottom = cell_top + self.CELL_SIZE
        
        # Map ball radius to grid scale
        scale_x = (self.GRID_SIZE * self.CELL_SIZE) / camera_width
        scale_y = (self.GRID_SIZE * self.CELL_SIZE) / camera_height
        grid_radius = ball_radius * max(scale_x, scale_y)
        
        # More relaxed tolerance for faster detection
        tolerance = 15  # Reduced tolerance for quicker response
        
        # Check if ball center is within cell bounds (more lenient)
        ball_left = grid_x - tolerance
        ball_right = grid_x + tolerance
        ball_top = grid_y - tolerance
        ball_bottom = grid_y + tolerance
        
        is_near_cell = (ball_left >= cell_left and 
                       ball_right <= cell_right and 
                       ball_top >= cell_top and 
                       ball_bottom <= cell_bottom)
        
        return is_near_cell, (row, col) if is_near_cell else None
    
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
    
    def handle_round_end(self, winner):
        """Handle the end of a round and determine if match continues"""
        if winner == Player.PLAYER1:
            self.player1_score += 1
            self.game_state = GameState.ROUND_PLAYER1_WON
            print(f"Round {self.current_round}: Player 1 wins! Score: {self.player1_score}-{self.player2_score}")
        elif winner == Player.PLAYER2:
            self.player2_score += 1
            self.game_state = GameState.ROUND_PLAYER2_WON
            print(f"Round {self.current_round}: Player 2 wins! Score: {self.player1_score}-{self.player2_score}")
        else:  # Draw
            self.game_state = GameState.ROUND_DRAW
            print(f"Round {self.current_round}: Draw! Score: {self.player1_score}-{self.player2_score}")
        
        # Check if match is over (best of 3)
        if self.player1_score == 2:
            self.game_state = GameState.MATCH_PLAYER1_WON
            print(f"MATCH OVER: Player 1 wins the match {self.player1_score}-{self.player2_score}!")
        elif self.player2_score == 2:
            self.game_state = GameState.MATCH_PLAYER2_WON
            print(f"MATCH OVER: Player 2 wins the match {self.player1_score}-{self.player2_score}!")
        elif self.current_round < self.max_rounds:
            # Start break before next round
            self.round_break_start_time = time.time()
            self.game_state = GameState.ROUND_BREAK
            print(f"15-second break before Round {self.current_round + 1}...")
        else:
            # All 3 rounds played, determine winner by score
            if self.player1_score > self.player2_score:
                self.game_state = GameState.MATCH_PLAYER1_WON
            elif self.player2_score > self.player1_score:
                self.game_state = GameState.MATCH_PLAYER2_WON
            else:
                # This shouldn't happen in best of 3, but just in case
                self.game_state = GameState.ROUND_DRAW
    
    def start_next_round(self):
        """Start the next round after break"""
        self.current_round += 1
        self.board = [[Player.NONE for _ in range(3)] for _ in range(3)]
        self.current_player = Player.PLAYER1
        self.game_state = GameState.PLAYING
        self.last_detection_time = 0
        self.position_history.clear()
        self.confirmation_counter = 0
        print(f"Round {self.current_round} starting!")
    
    def reset_match(self):
        """Reset the entire match"""
        self.board = [[Player.NONE for _ in range(3)] for _ in range(3)]
        self.current_player = Player.PLAYER1
        self.game_state = GameState.PLAYING
        self.last_detection_time = 0
        self.position_history.clear()
        self.confirmation_counter = 0
        self.player1_score = 0
        self.player2_score = 0
        self.current_round = 1
        self.round_break_start_time = 0
        print("New match started!")
    
    def show_input_method_selection(self):
        """Show input method selection screen"""
        print("Select your input method:")
        print("Press 'S' for Sword detection")
        print("Press 'B' for Ball detection")
        print("Press 'C' to calibrate ball detection")
        print("Press 'Q' to quit")
        
        while not self.input_method_selected:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Create selection screen
            selection_frame = frame.copy()
            
            # Add overlay
            overlay = np.zeros_like(selection_frame)
            overlay[:] = (0, 0, 0)
            selection_frame = cv2.addWeighted(selection_frame, 0.3, overlay, 0.7, 0)
            
            # Add text
            cv2.putText(selection_frame, "SELECT INPUT METHOD", 
                       (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.WHITE, 3)
            
            cv2.putText(selection_frame, "Press 'S' for GREEN SWORD detection", 
                       (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.GREEN, 2)
            
            cv2.putText(selection_frame, "Press 'B' for YELLOW SMILEY BALL detection", 
                       (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.CYAN, 2)
            
            cv2.putText(selection_frame, "Press 'C' to CALIBRATE ball detection", 
                       (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.ORANGE, 2)
            
            cv2.putText(selection_frame, "Press 'Q' to quit", 
                       (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.RED, 2)
            
            cv2.putText(selection_frame, "Recommended: Use YELLOW BALL for throwing, SWORD for pointing", 
                       (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.WHITE, 2)
            
            cv2.imshow('Input Method Selection', selection_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') or key == ord('S'):
                self.input_method = 'sword'
                self.input_method_selected = True
                print("Selected: Green Sword detection")
            elif key == ord('b') or key == ord('B'):
                self.input_method = 'ball'
                self.input_method_selected = True
                print("Selected: Yellow Smiley Ball detection")
            elif key == ord('c') or key == ord('C'):
                cv2.destroyWindow('Input Method Selection')
                self.run_calibration()
                # Reload calibration after running it
                self.load_ball_calibration()
                # Continue with selection
                continue
            elif key == ord('q') or key == ord('Q'):
                cv2.destroyAllWindows()
                sys.exit(0)
        
        cv2.destroyWindow('Input Method Selection')
    
    def run_calibration(self):
        """Run ball calibration from within the game"""
        print("Starting ball calibration...")
        self.cap.release()  # Release current camera
        
        # Import and run calibration
        try:
            import subprocess
            import sys
            result = subprocess.run([sys.executable, 'calibrate_ball.py'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("Calibration completed successfully!")
            else:
                print(f"Calibration error: {result.stderr}")
        except Exception as e:
            print(f"Error running calibration: {e}")
        
        # Restart camera
        self.setup_camera()
    
    def create_scoreboard_window(self):
        """Create a separate scoreboard window"""
        scoreboard_frame = np.ones((self.SCOREBOARD_HEIGHT, self.SCOREBOARD_WIDTH, 3), dtype=np.uint8) * 255
        
        # Title
        cv2.putText(scoreboard_frame, "MATCH SCOREBOARD", 
                   (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.BLACK, 2)
        
        # Round info
        cv2.putText(scoreboard_frame, f"Round {self.current_round} of {self.max_rounds}", 
                   (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.BLACK, 2)
        
        # Scores
        cv2.putText(scoreboard_frame, "PLAYER 1 (Red X)", 
                   (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.RED, 2)
        cv2.putText(scoreboard_frame, f"Wins: {self.player1_score}", 
                   (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.RED, 2)
        
        cv2.putText(scoreboard_frame, "PLAYER 2 (Blue O)", 
                   (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.BLUE, 2)
        cv2.putText(scoreboard_frame, f"Wins: {self.player2_score}", 
                   (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.BLUE, 2)
        
        # Match status
        if self.game_state == GameState.PLAYING:
            current_player = "Player 1" if self.current_player == Player.PLAYER1 else "Player 2"
            status_text = f"{current_player}'s Turn"
            color = self.RED if self.current_player == Player.PLAYER1 else self.BLUE
        elif self.game_state == GameState.ROUND_BREAK:
            remaining_time = int(self.round_break_duration - (time.time() - self.round_break_start_time))
            status_text = f"Break: {max(0, remaining_time)}s"
            color = self.ORANGE
        elif self.game_state == GameState.MATCH_PLAYER1_WON:
            status_text = "PLAYER 1 WINS!"
            color = self.RED
        elif self.game_state == GameState.MATCH_PLAYER2_WON:
            status_text = "PLAYER 2 WINS!"
            color = self.BLUE
        else:
            status_text = "Round Complete"
            color = self.BLACK
        
        cv2.putText(scoreboard_frame, status_text, 
                   (50, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return scoreboard_frame
   
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
                elif self.board[i][j] == Player.PLAYER1:
                    # Player 1: Red X
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
                    # Player 2: Blue O
                    radius = self.CELL_SIZE // 3
                    cv2.circle(grid_frame, (center_x, center_y), radius, self.BLUE, 8)
        
        # Game status
        if self.game_state == GameState.PLAYING:
            if self.current_player == Player.PLAYER1:
                text = "Player 1's Turn (Red X) - Use Green Sword or Yellow Ball!"
                color = self.RED
            else:
                text = "Player 2's Turn (Blue O) - Use Green Sword or Yellow Ball!"
                color = self.BLUE
        elif self.game_state == GameState.ROUND_PLAYER1_WON:
            text = f"Round {self.current_round}: Player 1 (Red X) Won!"
            color = self.RED
        elif self.game_state == GameState.ROUND_PLAYER2_WON:
            text = f"Round {self.current_round}: Player 2 (Blue O) Won!"
            color = self.BLUE
        elif self.game_state == GameState.ROUND_DRAW:
            text = f"Round {self.current_round}: It's a Draw!"
            color = self.BLACK
        elif self.game_state == GameState.ROUND_BREAK:
            remaining_time = int(self.round_break_duration - (time.time() - self.round_break_start_time))
            text = f"Break Time! Next round in {max(0, remaining_time)} seconds..."
            color = self.CYAN
        elif self.game_state == GameState.MATCH_PLAYER1_WON:
            text = "🏆 PLAYER 1 WINS THE MATCH! 🏆"
            color = self.RED
        elif self.game_state == GameState.MATCH_PLAYER2_WON:
            text = "🏆 PLAYER 2 WINS THE MATCH! 🏆"
            color = self.BLUE
        else:
            text = "Unknown state"
            color = self.BLACK
        
        cv2.putText(grid_frame, text, (50, 650), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, color, 3)
        
        # Progress bar (only for sword mode)
        if self.confirmation_counter > 0 and self.input_method == 'sword':
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
    

    
    def run(self):
        print("Two-Player AR Tic-Tac-Toe - BEST OF 3 MATCH!")
        
        # Show input method selection first
        self.show_input_method_selection()
        
        if self.input_method == 'sword':
            print("Using GREEN SWORD detection mode")
            print("- Point your GREEN SWORD at a cell")
            print("- Hold position steady for confirmation")
        else:
            print("Using YELLOW SMILEY BALL detection mode")
            print("- Throw YELLOW SMILEY BALL completely into a cell")
            print("- Ball will be detected instantly when it lands in a cell")
        
        print("\nThree windows will open:")
        print("1. 'Game Grid' - Cast this window to your TV")
        print("2. 'Camera Feed' - Shows detection")
        print("3. 'Scoreboard' - Shows match progress and scores")
        print("\nMatch Format:")
        print("- Best of 3 rounds (first to win 2 rounds wins the match)")
        print("- 15-second break between rounds")
        print("- Scoreboard tracks wins across rounds")
        print("- Press 'q' to quit, 'r' to restart match")
        
        while True:
            ret, camera_frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            # Don't flip camera frame - keep it natural
            camera_height, camera_width = camera_frame.shape[:2]
            
            grid_frame = self.create_grid_window()
            
            # Handle sword and ball detection for current player
            if self.game_state == GameState.PLAYING:
                
                current_time = time.time()
                
                # Detect based on selected input method
                if self.input_method == 'sword':
                    sword_detection = self.detect_green_sword(camera_frame)
                    ball_detection = None
                else:
                    sword_detection = None
                    ball_detection = self.detect_tennis_ball(camera_frame)
                
                # Handle ball detection (instant move, no confirmation needed)
                if ball_detection and self.input_method == 'ball':
                    move_made = self.process_ball_detection_instant(ball_detection, camera_width, camera_height)
                    
                    # Draw ball detection
                    cv2.drawContours(camera_frame, [ball_detection['contour']], -1, self.YELLOW, 2)
                    cv2.circle(camera_frame, ball_detection['center'], ball_detection['radius'], self.GREEN, 3)
                    cv2.circle(camera_frame, ball_detection['center'], 5, self.YELLOW, -1)
                    
                    # Show ball info
                    center = ball_detection['center']
                    cv2.putText(camera_frame, f"BALL (C:{ball_detection['circularity']:.2f} A:{int(ball_detection['area'])})", 
                               (center[0] + 20, center[1] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.CYAN, 2)
                    
                    # Check if ball is completely inside a cell
                    is_inside, cell_pos = self.is_ball_completely_in_cell(
                        ball_detection['center'], ball_detection['radius'], 
                        camera_width, camera_height)
                    
                    if is_inside and cell_pos:
                        row, col = cell_pos
                        if self.is_valid_move(row, col):
                            cv2.circle(camera_frame, center, ball_detection['radius'] + 15, self.GREEN, 4)
                            cv2.putText(camera_frame, f"VALID MOVE! Cell ({row},{col})", 
                                       (center[0] + 30, center[1]), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.GREEN, 2)
                        else:
                            cv2.putText(camera_frame, "Cell occupied!", 
                                       (center[0] + 30, center[1] + 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.RED, 2)
                    else:
                        cv2.putText(camera_frame, "Ball not fully in cell!", 
                                   (center[0] + 30, center[1] + 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.ORANGE, 2)
                
                elif sword_detection and self.input_method == 'sword':
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
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.CYAN, 2)
                                
                                self.confirmation_counter += 1
                                
                                if self.confirmation_counter >= self.required_confirmations:
                                    if current_time - self.last_detection_time > self.detection_cooldown:
                                        if self.make_move(row, col, self.current_player):
                                            player_name = "Player 1" if self.current_player == Player.PLAYER1 else "Player 2"
                                            print(f"{player_name} played at ({row}, {col}) with GREEN SWORD")
                                            self.last_detection_time = current_time
                                            self.confirmation_counter = 0
                                            self.position_history.clear()
                                            
                                            winner = self.check_winner()
                                            if winner == Player.PLAYER1 or winner == Player.PLAYER2 or winner == "DRAW":
                                                self.handle_round_end(winner)
                                            else:
                                                # Switch to other player
                                                self.current_player = Player.PLAYER2 if self.current_player == Player.PLAYER1 else Player.PLAYER1
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
                    if self.input_method == 'sword':
                        cv2.putText(camera_frame, "Show green sword!", 
                                   (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.RED, 2)
                    else:
                        cv2.putText(camera_frame, "Throw yellow smiley ball into a cell!", 
                                   (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.CYAN, 2)
                        
                        # Debug: Show all detected circles for troubleshooting
                        self.debug_show_all_circles(camera_frame)
            
            # Handle round break timer
            elif self.game_state == GameState.ROUND_BREAK:
                elapsed_time = time.time() - self.round_break_start_time
                if elapsed_time >= self.round_break_duration:
                    self.start_next_round()
                else:
                    remaining_time = int(self.round_break_duration - elapsed_time)
                    cv2.putText(camera_frame, f"Break Time! Next round in {remaining_time}s", 
                               (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.ORANGE, 2)
            
            # Instructions
            if self.game_state == GameState.PLAYING:
                current_player_name = "Player 1 (X)" if self.current_player == Player.PLAYER1 else "Player 2 (O)"
                if self.input_method == 'sword':
                    instructions = [
                        f"Round {self.current_round}/3 - {current_player_name}'s turn:",
                        "Point GREEN SWORD at cell",
                        "Hold steady for confirmation",
                        "Press 'q' to quit, 'r' to restart match"
                    ]
                else:
                    instructions = [
                        f"Round {self.current_round}/3 - {current_player_name}'s turn:",
                        "Throw YELLOW SMILEY BALL into cell",
                        "Ball detected instantly when in cell",
                        "Press 'q' to quit, 'r' to restart match"
                    ]
            elif self.game_state in [GameState.MATCH_PLAYER1_WON, GameState.MATCH_PLAYER2_WON]:
                instructions = [
                    "MATCH COMPLETE!",
                    f"Final Score: {self.player1_score}-{self.player2_score}",
                    "",
                    "Press 'r' to start new match",
                    "Press 'q' to quit"
                ]
            else:
                instructions = [
                    f"Round {self.current_round}/3",
                    f"Score: Player 1: {self.player1_score} | Player 2: {self.player2_score}",
                    "",
                    "Press 'q' to quit, 'r' to restart match",
                    ""
                ]
            
            for i, instruction in enumerate(instructions):
                cv2.putText(camera_frame, instruction, (10, 30 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.WHITE, 1)
            
            # Show progress (only for sword mode)
            if self.confirmation_counter > 0 and self.input_method == 'sword':
                progress_text = f"Confirming: {int((self.confirmation_counter/self.required_confirmations)*100)}%"
                cv2.putText(camera_frame, progress_text, (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.CYAN, 2)
            
            # Show detection info in corner
            if 'sword_detection' in locals() and sword_detection:
                mask_small = cv2.resize(sword_detection['mask'], (160, 120))
                mask_colored = cv2.applyColorMap(mask_small, cv2.COLORMAP_JET)
                camera_frame[10:130, camera_width-170:camera_width-10] = mask_colored
                cv2.putText(camera_frame, "Sword", (camera_width-165, 145), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.WHITE, 1)
            
            if 'ball_detection' in locals() and ball_detection:
                # For ball detection, show a simple circle indicator instead of mask
                cv2.rectangle(camera_frame, (camera_width-170, 140), (camera_width-10, 260), self.BLACK, 2)
                cv2.rectangle(camera_frame, (camera_width-168, 142), (camera_width-12, 258), self.WHITE, -1)
                
                # Draw detected ball info
                center_x, center_y = camera_width-90, 200
                cv2.circle(camera_frame, (center_x, center_y), 30, self.GREEN, 3)
                cv2.circle(camera_frame, (center_x, center_y), 5, self.GREEN, -1)
                cv2.putText(camera_frame, "Ball Detected", (camera_width-165, 275), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.BLACK, 1)
            
            # Create and show all windows
            scoreboard_frame = self.create_scoreboard_window()
            
            cv2.imshow('Game Grid (Cast this to TV)', grid_frame)
            cv2.imshow('Camera Feed - Detection', camera_frame)
            cv2.imshow('Scoreboard', scoreboard_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.reset_match()
                print("Match reset!")
        
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