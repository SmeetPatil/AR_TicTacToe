import cv2
import sys

def list_cameras():
    """List all available cameras and let user select one"""
    print("Scanning for available cameras...")
    available_cameras = []
    
    # Test cameras 0-20 to find all available ones
    for i in range(21):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                height, width = frame.shape[:2]
                print(f"Camera {i}: Available - Resolution: {width}x{height}")
                available_cameras.append(i)
                
                # Show a preview of this camera
                cv2.putText(frame, f"Camera {i} - Press any key to continue", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow(f'Camera {i} Preview', frame)
                cv2.waitKey(2000)  # Show for 2 seconds
                cv2.destroyAllWindows()
            cap.release()
        else:
            print(f"Camera {i}: Not available")
    
    if not available_cameras:
        print("No cameras found!")
        return None
    
    print(f"\nFound {len(available_cameras)} camera(s): {available_cameras}")
    
    # Let user select camera
    while True:
        try:
            choice = input(f"Select camera index {available_cameras}: ")
            camera_index = int(choice)
            if camera_index in available_cameras:
                return camera_index
            else:
                print(f"Invalid choice. Please select from {available_cameras}")
        except ValueError:
            print("Please enter a valid number")

def test_selected_camera(camera_index):
    """Test the selected camera with live preview"""
    print(f"Testing camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Failed to open camera {camera_index}")
        return False
    
    print("Camera opened successfully!")
    print("Press 'q' to quit, 's' to save this camera choice")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        # Add instructions
        cv2.putText(frame, f"Camera {camera_index} - Press 'q' to quit, 's' to select", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Move your hand to test detection", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Camera Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return False
        elif key == ord('s'):
            cap.release()
            cv2.destroyAllWindows()
            return True
    
    cap.release()
    cv2.destroyAllWindows()
    return False

def main():
    print("Camera Selection Tool")
    print("=" * 30)
    
    # List and select camera
    selected_camera = list_cameras()
    
    if selected_camera is None:
        print("No camera selected. Exiting.")
        return
    
    # Test selected camera
    if test_selected_camera(selected_camera):
        print(f"\nCamera {selected_camera} selected!")
        
        # Save camera choice to a config file
        with open('camera_config.txt', 'w') as f:
            f.write(str(selected_camera))
        
        print(f"Camera index {selected_camera} saved to camera_config.txt")
        print("You can now run the AR Tic-Tac-Toe game!")
    else:
        print("Camera not selected. Run this tool again to choose a different camera.")

if __name__ == "__main__":
    main()