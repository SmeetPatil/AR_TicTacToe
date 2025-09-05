"""
Setup script for AR Tic-Tac-Toe
Run this to install dependencies and test your camera setup
"""

import subprocess
import sys
import cv2

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing packages: {e}")
        return False

def test_camera():
    """Test camera connection"""
    print("\nTesting camera connection...")
    
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✓ Camera {i} is working!")
                print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
                cap.release()
                return i
            cap.release()
    
    print("✗ No working camera found")
    print("Make sure DroidCam is running and connected to your PC")
    return None

def main():
    print("AR Tic-Tac-Toe Setup")
    print("=" * 30)
    
    # Install requirements
    if not install_requirements():
        return
    
    # Test camera
    camera_index = test_camera()
    
    if camera_index is not None:
        print(f"\n✓ Setup complete! Camera found at index {camera_index}")
        print("\nNext steps:")
        print("1. Make sure you have a red ball to play with")
        print("2. Run: python ar_tictactoe.py")
        print("3. Position your camera to see the projected grid")
        print("4. Have fun playing!")
    else:
        print("\n⚠ Camera setup incomplete")
        print("Please check your DroidCam connection and try again")

if __name__ == "__main__":
    main()