
# Shape Analysis Project

## Overview
This project focuses on analyzing and predicting geometric shapes using their features. It combines data processing, machine learning, and visualization to enable automated shape recognition and classification. The key components include:

- **Dataset Analysis**: A dataset of shapes with attributes such as area, perimeter, and compactness.
- **Shape Prediction**: Machine learning models for predicting shape categories.
- **Visualization**: Tools for analyzing shapes visually using their features.


## Usage

### Video Demonstration
Watch a demonstration of the project output:

**click on image to see video**



[![Shape Analysis Video](https://github.com/user-attachments/assets/a565fb26-8ad1-4b77-80d4-0f60a9e3f431)](https://github.com/prajwalsable99/-Shape-Analysis-Project-/blob/main/output.mp4)


## Directory Structure
```
SHAPE-DETECTION
|-- .ipynb_checkpoints
|-- shapes
|   |-- circle
|   |-- square
|   |-- star
|   |-- triangle
|   |-- circle-test.png
|-- Shape-det-App.ipynb
|-- Shape-Prediction.ipynb
|-- shapes_dataset_with_additional_features.csv
|-- square-test.png
|-- supervised-model.pkl
|-- test-star.jfif
|-- test-traingle.jfif
```

## Files and Structure

### 1. **Shape-Prediction.ipynb**
This notebook is responsible for creating the dataset and training the predictive model. It includes:
- Reading shape images from the `shapes` folder.
- Extracting features such as area, perimeter, aspect ratio, compactness, solidity, and convexity.
- Building and evaluating a machine learning model to predict shape categories.
- Saving the trained model for use in the `Shape-det-App.ipynb` notebook.

### 2. **shapes_dataset_with_additional_features.csv**
This dataset includes 14,970 entries with attributes:
- **Image Name**: Identifier for the shape image.
- **Shape Name**: Category of the shape (e.g., Circle, Triangle).
- **Sides**: Number of sides.
- **Area and Perimeter**: Geometric properties of the shape.
- **Aspect Ratio, Compactness, Solidity, Convexity**: Additional derived features for analysis.

### 3. **Shape-det-App.ipynb**
This notebook contains code for visualizing and detecting shapes in images using a pre-trained model. It includes:
- Integration with OpenCV for real-time shape detection.
- Utilization of a model trained on a custom dataset for shape prediction.
- Feature extraction and visualization of results.

## Installation and Requirements
To run the project, ensure you have the following:

- **Python** (3.8 or above)
- Required libraries:
  ```bash
  pip install numpy pandas matplotlib opencv-python scikit-learn
  ```




### Running the Notebooks
1. Open `Shape-Prediction.ipynb` to create the dataset and train the predictive model. Ensure the `shapes` folder containing the shape images is available.
2. Use `Shape-det-App.ipynb` to load the trained model and perform real-time shape detection and analysis.

### Exploring the Dataset
Load and explore the dataset to understand shape properties and their relationships:
```python
import pandas as pd

data = pd.read_csv('shapes_dataset_with_additional_features.csv')
print(data.head())
```

### Extending the Project
- Add more image processing techniques to improve feature extraction.
- Experiment with additional machine learning models.
- Deploy the application for real-time shape detection.

## Contributing
Feel free to contribute to the project by:
- Improving feature extraction techniques.
- Enhancing machine learning models.
- Adding more visualization tools.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
