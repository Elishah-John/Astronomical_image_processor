# Astronomical Image Processor

## Overview

The **Astronomical Image Processor** is a Python-based GUI application that helps astronomers and researchers analyze celestial images efficiently. It provides tools for detecting objects, measuring circularity, and enhancing image clarity, making it easier to extract meaningful insights from astronomical data. It provides tools for **single-image processing, batch processing, and visualization**, allowing users to extract useful information such as object contours, circularity, edge density, and image entropy. The application is built using **Tkinter, OpenCV, Matplotlib, and SciPy**.

## Features

- **Single Image Processing:**
  - Load and display astronomical images.
  - Apply different colormap visualizations.
  - Detect object contours and analyze circularity.
  - View image intensity statistics.
- **Batch Processing:**
  - Process multiple images from a folder.
  - Extract statistical data such as dimensions, file size, mean intensity, and circular object count.
  - Save batch processing results as a CSV file.
- **Advanced Visualization:**
  - Histogram analysis with options for log scaling, mean, median, and standard deviation display.
  - Edge detection using adjustable threshold values.
  - Interactive Region of Interest (ROI) selection.

## Installation

### Prerequisites

Ensure you have Python **3.7+** installed along with the following dependencies:

```bash
pip install numpy pandas matplotlib opencv-python-headless Pillow scipy
```

### Running the Program

Run the script using:

```bash
python batch_image_count.py
```

## Usage Guide

1. **Launch the Application** and select an image using the "Load Image" button.
2. Use the **Single Image Processing** tab to analyze contours, circularity, and intensity statistics.
3. Switch to the **Batch Processing** tab to process multiple images at once.
4. Use the **Visualization** tab to analyze histograms, edge detection, and ROI selection.
5. Save batch results to a CSV file for further analysis.

## File Structure

```
📂 Astronomical Image Processor
│── batch_image_count.py  # Main Python script
│── app_icon.png          # Application icon (optional)
│── batch_results.csv     # Generated results from batch processing
```

## License

This project is licensed under the **MIT License**.

## Author

Developed by Elishah John. Feel free to contribute or report issues!

