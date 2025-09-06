#!/usr/bin/env python3
"""
Windows-compatible ball calibration script
"""

import cv2
import numpy as np
import json
import time
from collections import deque

def main():
    print("=== YELLOW SMILEY BALL CALIBRATION ===")
    print("Instructions:")
    print("1. Hold your yellow smiley ball in front of the camera")
    print("2. Adjust HSV sliders until only your ball is highlighted")
    print("3. Press SPACE to start collecting samples")
    print("4. Move the ball around for 5 seconds")
    print("5. Press 's' to save calibration")
    print("6. Press 'q' to quit")
    
    # Setup camera
    camera_index = 0
    try:
        with open('camera_config.txt', 'r') as f:
            camera_index = int(f.read().strip())
    except:
        pass
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("[ERROR] Could not open camera")
        return False
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # HSV adjustment trackbars
    cv2.namedWindow('Ball Calibration')
    cv2.createTrackbar('H_min', 'Ball Calibration', 10, 179, lambda x: None)
    cv2.createTrackbar('S_min', 'Ball Calibration', 80, 255, lambda x: None)
    cv2.createTrackbar('V_min', 'Ball Calibration', 80, 255, lambda x: None)
    cv2.createTrackbar('H_max', 'Ball Calibration', 40, 179, lambda x: None)
    cv2.createTrackbar('S_max', 'Ball Calibration', 255, 255, lambda x: None)
    cv2.createTrackbar('V_max', 'Ball Calibration', 255, 255, lambda x: None)
    
    samples = []
    is_calibrating = False
    calibration_complete = False
    calibration_start_time = 0
    sample_duration = 5.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Get trackbar values
        h_min = cv2.getTrackbarPos('H_min', 'Ball Calibration')
        s_min = cv2.getTrackbarPos('S_min', 'Ball Calibration')
        v_min = cv2.getTrackbarPos('V_min', 'Ball Calibration')
        h_max = cv2.getTrackbarPos('H_max', 'Ball Calibration')
        s_max = cv2.getTrackbarPos('S_max', 'Ball Calibration')
        v_max = cv2.getTrackbarPos('V_max', 'Ball Calibration')
        
        # Create mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Clean mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw detections
        result = frame.copy()
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 200:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                cv2.drawContours(result, [contour], -1, (0, 255, 255), 2)
                cv2.circle(result, center, radius, (0, 255, 0), 2)
                cv2.circle(result, center, 3, (0, 0, 255), -1)
                
                cv2.putText(result, f"A:{int(area)} C:{circularity:.2f}", 
                           (center[0] + radius + 5, center[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Collect samples during calibration
                if is_calibrating and circularity > 0.3:
                    sample = {
                        'area': area,
                        'circularity': circularity,
                        'radius': radius,
                        'hsv_range': [[h_min, s_min, v_min], [h_max, s_max, v_max]]
                    }
                    samples.append(sample)
        
        # Calibration status
        current_time = time.time()
        if is_calibrating:
            elapsed = current_time - calibration_start_time
            remaining = sample_duration - elapsed
            
            if remaining > 0:
                cv2.putText(result, f"COLLECTING SAMPLES: {remaining:.1f}s", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(result, f"Samples: {len(samples)}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                is_calibrating = False
                calibration_complete = True
                print(f"[OK] Collected {len(samples)} samples")
        
        if calibration_complete:
            cv2.putText(result, "CALIBRATION COMPLETE! Press 's' to save", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        else:
            cv2.putText(result, "Adjust sliders, then press SPACE to calibrate", 
                       (10, result.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show HSV range
        cv2.putText(result, f"HSV: [{h_min},{s_min},{v_min}] - [{h_max},{s_max},{v_max}]", 
                   (10, result.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.imshow('Ball Calibration', result)
        cv2.imshow('Yellow Mask', mask)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and not is_calibrating and not calibration_complete:
            is_calibrating = True
            samples = []
            calibration_start_time = time.time()
            print("[INFO] Starting sample collection...")
        elif key == ord('s') and calibration_complete:
            # Save calibration
            if len(samples) >= 10:
                # Analyze samples
                areas = [s['area'] for s in samples]
                circularities = [s['circularity'] for s in samples]
                
                area_mean = np.mean(areas)
                area_std = np.std(areas)
                circ_mean = np.mean(circularities)
                circ_std = np.std(circularities)
                
                calibration_data = {
                    'yellow_lower': [h_min, s_min, v_min],
                    'yellow_upper': [h_max, s_max, v_max],
                    'min_area': max(200, int(area_mean - 2 * area_std)),
                    'max_area': int(area_mean + 3 * area_std),
                    'min_circularity': max(0.3, circ_mean - 2 * circ_std)
                }
                
                try:
                    with open('ball_calibration.json', 'w') as f:
                        json.dump(calibration_data, f, indent=2)
                    
                    print("[OK] Calibration saved!")
                    print(f"HSV range: {calibration_data['yellow_lower']} - {calibration_data['yellow_upper']}")
                    print(f"Area range: {calibration_data['min_area']} - {calibration_data['max_area']}")
                    print(f"Min circularity: {calibration_data['min_circularity']:.2f}")
                    break
                except Exception as e:
                    print(f"[ERROR] Could not save calibration: {e}")
            else:
                print("[ERROR] Not enough samples collected!")
        elif key == ord('r'):
            is_calibrating = False
            calibration_complete = False
            samples = []
            print("[INFO] Calibration restarted")
        elif key == ord('q'):
            print("[INFO] Calibration cancelled")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return calibration_complete

if __name__ == "__main__":
    success = main()
    if success:
        print("\n[OK] Calibration complete! You can now run the main game.")
    else:
        print("\n[INFO] Calibration was not completed.")