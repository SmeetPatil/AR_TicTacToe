# AR Tic-Tac-Toe

An augmented reality tic-tac-toe game where you play by throwing a yellow smiley ball into grid cells.

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install opencv-python numpy
   ```

2. **Test ball detection**:
   ```bash
   python quick_test.py
   ```

3. **Calibrate your yellow ball** (if needed):
   ```bash
   python calibrate_ball.py
   ```

4. **Run the game**:
   ```bash
   python ar_tictactoe.py
   ```

## How to Play

1. Select 'B' for ball detection mode
2. Throw your yellow smiley ball into any empty grid cell
3. Game alternates between Player 1 (Red X) and Player 2 (Blue O)
4. First to get 3 in a row wins the round
5. Best of 3 rounds wins the match

## Files

- `ar_tictactoe.py` - Main game
- `calibrate_ball.py` - Ball calibration tool
- `quick_test.py` - Test if ball detection works
- `camera_selector.py` - Camera setup utility
- `setup.py` - Initial setup script

## Troubleshooting

**Ball not detected?**
1. Run `python quick_test.py` to check basic detection
2. Run `python calibrate_ball.py` to fine-tune settings
3. Ensure good lighting and clear background

**Camera issues?**
- Run `python camera_selector.py` to find your camera
- Make sure no other apps are using the camera

## Controls

- **'q'**: Quit
- **'r'**: Restart match
- **'c'**: Calibrate ball (from main menu)

## Contributors 
- ### Smeet Patil
