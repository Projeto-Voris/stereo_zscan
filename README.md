# Stereo Z-Scan

## Overview

**Stereo Z-Scan** is a project designed to perform high-resolution 3D reconstruction using stereo camera systems. This repository provides the tools and algorithms necessary for capturing and processing stereo images, enabling users to generate detailed 3D models from 2D image pairs.

## Features

- **Stereo Image Capture**: Interfaces with stereo cameras to capture synchronized image pairs.
- **3D Reconstruction**: Implements algorithms for depth estimation and 3D point cloud generation.
- **Visualization Tools**: Provides utilities for visualizing 3D reconstructions and analyzing the results.
- **CUDA Acceleration**: Utilizes GPU processing for improved performance in depth estimation and 3D modeling.

## Requirements

- **Hardware**: 
  - Stereo camera system (e.g., Spinnaker camera)
  - GPU with CUDA support (recommended for acceleration)

- **Software**:
  - Python 3.x
  - OpenCV 4.x
  - (Spinnaker SDK for Python)[https://www.teledynevisionsolutions.com/products/spinnaker-sdk/?model=Spinnaker%20SDK&vertical=machine%20vision&segment=iis]

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Projeto-Voris/stereo_zscan.git
   cd stereo_zscan
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your camera according to the manufacturer's instructions.

## Usage


Contributions are welcome! Please feel free to submit issues or pull requests. For major changes, please open an issue first to discuss what you would like to change.


