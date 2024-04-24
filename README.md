# Real-time Object Detection

Welcome to Real-time Object Detection! This project utilizes the power of machine learning to detect objects in real-time using a pre-trained SSD MobileNet V3 model. With this project, you can easily identify various objects, such as people, vehicles, animals, and more, directly from your webcam feed.

<p align="center">
  <img src="gifs/crowd.gif" alt="Crowd">
</p>

## How It Works

1. **Model Initialization**: The project loads a pre-trained SSD MobileNet V3 model, which has been trained on the COCO dataset for object detection.

2. **Object Detection**: Using the model, the project performs real-time object detection on the frames captured from the webcam feed.

3. **Display Results**: Detected objects are highlighted with bounding boxes and labeled with their corresponding class names in real-time, providing instant feedback.

<p align="center">
  <img src="gifs/zebra.gif" alt="Zebra">
</p>

## Getting Started

To run Real-time Object Detection on your machine, follow these simple steps:

1. **Clone the Repository**: Use the following command to clone the repository to your local machine:

```bash
git clone https://github.com/KelvinPuyam/Real-time-object-detection
```

2. **Navigate to the Project Directory**: Change your current directory to the cloned repository:

```bash
cd Real-time-object-detection
```

3. **Run the Script**: Execute the `real_time_object_detection.py` script using Python:

```bash
python real_time_object_detection.py
```

4. **Enjoy Real-time Object Detection**: Once the script is running, your webcam will activate, and you'll see objects being detected in real-time on your screen. Press 'q' to quit the application and you will find a video recording of your detection session saved in the project directory. Happy detecting! 🎥🔍

<p align="center">
  <img src="gifs/plane.gif" alt="Plane">
</p>

## Requirements

- Python 3
- OpenCV

## Contributions

Contributions are welcome! Fork the repository, make changes in a new branch, and submit a pull request (PR). Provide a clear description of your changes. PRs will be reviewed by maintainers before merging. Feel free to [open an issue](https://github.com/KelvinPuyam/Real-time-object-detection/issues) for suggestions or questions. Thank you for contributing! 🚀

<p align="center">
  <img src="gifs/interview.gif" alt="Interview">
</p>
