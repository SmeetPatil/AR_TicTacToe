#!/usr/bin/env python3
"""
Quick test to check if ball detection is working
"""

import cv2
import numpy as np

def quick_test():
    print("=== QUICK BALL DETECTION TEST ===")
    print("This will show you what the camera sees and if yellow objects are detected")
    print("Press 'q' to quit")
    
    # Setup camera
    camera_index = 0
    try:
        with open('camera_config.txt', 'r') as f:
            camera_index = int(f.read().strip())
    except:
        pass
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Very permissive yellow detection
    yellow_lower = np.array([10, 80, 80])
    yellow_upper = np.array([40, 255, 255])
    
    print(f"Looking for yellow objects with HSV range: {yellow_lower} - {yellow_upper}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask
        mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw all yellow objects
        result = frame.copy()
        detection_count = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 200:  # Minimum area
                detection_count += 1
                
                # Get circle
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                # Draw detection
                cv2.drawContours(result, [contour], -1, (0, 255, 255), 2)
                cv2.circle(result, center, radius, (0, 255, 0), 2)
                cv2.circle(result, center, 3, (0, 0, 255), -1)
                
                # Show info
                cv2.putText(result, f"Area: {int(area)}", 
                           (center[0] + radius + 5, center[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(result, f"Circ: {circularity:.2f}", 
                           (center[0] + radius + 5, center[1] + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Show status
        cv2.putText(result, f"Yellow objects detected: {detection_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if detection_count > 0:
            cv2.putText(result, "DETECTION WORKING!", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(result, "No yellow objects found", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(result, "Try adjusting lighting or ball position", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Show frames
        cv2.imshow('Quick Test - Camera Feed', result)
        cv2.imshow('Quick Test - Yellow Mask', mask)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if detection_count > 0:
        print("\n[OK] Detection is working! Your ball should be detected in the main game.")
        print("If it's still not working in the game, try running: python calibrate_ball.py")
    else:
        print("\n[ERROR] No yellow objects detected.")
        print("Try:")
        print("1. Better lighting")
        print("2. Different ball position")
        print("3. Run: python calibrate_ball.py to fine-tune settings")

if __name__ == "__main__":
    quick_test()