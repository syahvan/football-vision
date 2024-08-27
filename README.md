# Football Analysis with Computer Vision

## Overview

This project aims to enhance football game analysis by utilizing cutting-edge computer vision techniques to detect and track players, referees, and the ball within video footage. The YOLO model and K-Means clustering are employed to distinguish players by their jersey colors, assign them to teams, and compute ball possession percentages. Optical flow techniques are used to measure camera movement between frames, ensuring precise tracking of player motion. Additionally, perspective transformation converts pixel measurements into real-world meters, enabling accurate calculations of speed and distance covered. By integrating these advanced methods, the project provides comprehensive insights into game dynamics and player performance, offering valuable tools for in-depth sports analytics.

![Screenshot](output_videos/output_videos.gif)

## Key Features

- **Player and Ball Detection:** Utilizes YOLOv8 for detecting and tracking players and the ball in video frames.
- **Color Segmentation:** Applies K-Means clustering for pixel segmentation to differentiate player jersey colors.
- **Motion Analysis:** Measures camera and player movements using Optical Flow for dynamic game analysis.
- **Perspective Transformation:** Adjusts for camera angles to accurately represent scene depth and spatial relations.
- **Speed and Distance Calculation:** Computes player speed and distance for performance metrics.

## Trained Models

The trained YOLOv8 model used for detecting players and the ball can be downloaded from [here](https://drive.google.com/file/d/1gRTMjSiqf4gZ-LvRqZlIgg1CUnEhhadK/view?usp=sharing)

## Requirements

To run this project, ensure you have Python 3.x installed and install the required libraries using:

```bash
pip install ultralytics supervision opencv-python numpy matplotlib pandas
```

## Usage

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/syahvan/football-analysis.git
   cd football-analysis
   ```

2. **Download the YOLOv8 Model:**

   Place the YOLOv8 model in the `models` directory or specify the path in the configuration.

3. **Run the Analysis Script:**

   ```bash
   python analyze_football.py --input_video path_to_your_video --model_path path_to_your_model
   ```

   Replace `path_to_your_video` with the path to your input video and `path_to_your_model` with the path to the YOLOv8 model file.

4. **Review Results:**

   Processed videos and analysis results will be saved in the `output_videos` directory. Check this directory for the results of the analysis.

## References

- [ Build an AI/ML Football Analysis system with YOLO, OpenCV, and Python](https://youtu.be/neBZ6huolkg?si=RAYHOSGn3JWGavbb)