import cv2
import numpy as np

def create_test_frame():
    """Create a test frame to show what the grid looks like"""
    # Create a blank frame (640x480, white background)
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Grid settings
    GRID_SIZE = 3
    CELL_SIZE = 150
    GRID_OFFSET_X = 100
    GRID_OFFSET_Y = 100
    BLACK = (0, 0, 0)
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    
    # Draw grid lines
    for i in range(4):
        # Vertical lines
        x = GRID_OFFSET_X + i * CELL_SIZE
        cv2.line(frame, (x, GRID_OFFSET_Y), 
                (x, GRID_OFFSET_Y + 3 * CELL_SIZE), 
                BLACK, 3)
        
        # Horizontal lines
        y = GRID_OFFSET_Y + i * CELL_SIZE
        cv2.line(frame, (GRID_OFFSET_X, y), 
                (GRID_OFFSET_X + 3 * CELL_SIZE, y), 
                BLACK, 3)
    
    # Draw some example X and O symbols
    # X in top-left (0,0)
    center_x = GRID_OFFSET_X + 0 * CELL_SIZE + CELL_SIZE // 2
    center_y = GRID_OFFSET_Y + 0 * CELL_SIZE + CELL_SIZE // 2
    offset = CELL_SIZE // 3
    cv2.line(frame, 
            (center_x - offset, center_y - offset),
            (center_x + offset, center_y + offset),
            RED, 5)
    cv2.line(frame, 
            (center_x + offset, center_y - offset),
            (center_x - offset, center_y + offset),
            RED, 5)
    
    # O in center (1,1)
    center_x = GRID_OFFSET_X + 1 * CELL_SIZE + CELL_SIZE // 2
    center_y = GRID_OFFSET_Y + 1 * CELL_SIZE + CELL_SIZE // 2
    radius = CELL_SIZE // 3
    cv2.circle(frame, (center_x, center_y), radius, BLUE, 5)
    
    # Add title
    cv2.putText(frame, "AR Tic-Tac-Toe Grid Preview", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLACK, 2)
    
    # Add instructions
    instructions = [
        "This is how the grid will look in the game",
        "Red X = Human player",
        "Blue O = AI player",
        "Press any key to close"
    ]
    
    for i, instruction in enumerate(instructions):
        cv2.putText(frame, instruction, (10, 400 + i * 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 1)
    
    return frame

def main():
    print("Showing grid preview...")
    frame = create_test_frame()
    
    cv2.imshow('Grid Preview', frame)
    cv2.waitKey(0)  # Wait for any key press
    cv2.destroyAllWindows()
    print("Grid preview closed.")

if __name__ == "__main__":
    main()