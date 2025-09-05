# AR Tic-Tac-Toe

An augmented reality tic-tac-toe game where you play by hitting grid cells with a colored ball while the camera tracks your moves.

## Features

- Real-time ball detection using computer vision
- AI opponent with strategic gameplay
- Works with phone camera via DroidCam
- Customizable ball colors and detection parameters
- Visual feedback and game state display

## Requirements

- Python 3.7+
- A colored ball (red works best by default)
- Camera (phone via DroidCam or webcam)
- Good lighting conditions

## Setup

1. **Install DroidCam** (if using phone camera):
   - Download DroidCam on your phone and PC
   - Connect your phone to PC via USB or WiFi
   - Start DroidCam on both devices

2. **Install dependencies**:
   ```bash
   python setup.py
   ```

3. **Test your setup**:
   The setup script will test your camera connection and install required packages.

## How to Play

1. **Start the game**:
   ```bash
   python ar_tictactoe.py
   ```

2. **Position your camera** so it can see the projected grid on your screen

3. **Play your turn** by hitting an empty grid cell with your ball

4. **Wait for AI** to make its move

5. **Continue** until someone wins or it's a draw

## Controls

- `q` - Quit the game
- `r` - Restart the game

## Configuration

Edit `config.py` to customize:

- **Ball color detection**: Adjust HSV color ranges for different colored balls
- **Grid size and position**: Change cell size and grid offset
- **Detection sensitivity**: Modify ball radius constraints
- **Timing**: Adjust detection cooldown and AI delay

### Ball Color Calibration

If the default red ball detection doesn't work well:

1. Use an HSV color picker tool to find your ball's HSV values
2. Update `BALL_COLOR_LOWER` and `BALL_COLOR_UPPER` in `config.py`
3. Test with different lighting conditions

Common color ranges:
- **Red**: `[0, 100, 100]` to `[10, 255, 255]`
- **Blue**: `[100, 100, 100]` to `[130, 255, 255]`
- **Green**: `[40, 100, 100]` to `[80, 255, 255]`
- **Yellow**: `[20, 100, 100]` to `[30, 255, 255]`

## Troubleshooting

### Camera Issues
- Make sure DroidCam is running on both phone and PC
- Try different camera indices (0, 1, 2) if camera doesn't connect
- Check that no other applications are using the camera

### Ball Detection Issues
- Ensure good lighting conditions
- Use a solid-colored ball with good contrast against background
- Adjust HSV color ranges in config.py
- Make sure ball size is within the configured radius range

### Performance Issues
- Close other applications using the camera
- Reduce camera resolution in config.py
- Ensure stable connection between phone and PC

## Game Rules

- Human player uses X (red)
- AI player uses O (blue)
- First to get 3 in a row (horizontal, vertical, or diagonal) wins
- If all cells are filled without a winner, it's a draw

## Technical Details

The game uses:
- **OpenCV** for computer vision and ball detection
- **HSV color space** for robust color detection
- **Morphological operations** to clean up detection noise
- **Simple AI** with win/block/strategic positioning logic

Enjoy playing AR Tic-Tac-Toe!