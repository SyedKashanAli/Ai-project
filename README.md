# COVID-19 Prediction from Chest X-ray Images Using CNN

## Project Overview
This project involves building and training a **Convolutional Neural Network (CNN)** in Keras with TensorFlow as the backend to predict whether patients are infected with COVID-19 based on their chest X-ray images. The dataset contains X-ray images of both COVID-infected and non-infected patients.

Key highlights of the project:
- **Frameworks Used**: TensorFlow 2.0, Keras
- **Data Visualization**: Matplotlib
- **Data Source**: Dataset cloned from a GitHub repository "https://github.com/RishitToteja/Chext-X-ray-Images-Data-Set.git".


---

## Model Architecture
The CNN was implemented using the Sequential API in Keras. Below is a summary of the architecture:
1. **Convolutional Layers**: Extract spatial features from the X-ray images.
2. **Pooling Layers**: Reduce spatial dimensions to prevent overfitting.
3. **Dropout Layers**: Add regularization to avoid overfitting.
4. **Dense Layers**: Perform the final classification.
5. **Activation Functions**:
   - `relu` for hidden layers
   - `sigmoid` for the output layer

### Optimizer and Learning Rate
- **Optimizer**: Adam
- **Learning Rate**: 0.001

---

## Data Preprocessing and Augmentation
- Preprocessing and augmentation were carried out using **TensorFlow 2.0** to improve model generalization.
- The dataset was divided into three subsets:
  - **Training Set**
  - **Validation Set**
  - **Test Set**

---

## Results
The model's performance was evaluated using the following metrics:

| Metric                  | Training Set | Validation Set | Test Set |
|-------------------------|--------------|----------------|----------|
| **Accuracy**            | 98.41%      | 97.51%         | 94.83%   |
| **Loss**                | 0.0490      | 0.0759         | 0.1141   |

---

## Installation
Follow these steps to set up the project:

### 1. Clone the Dataset
Use the following command in your Python environment to clone the dataset from the GitHub repository:
```bash
!git clone https://github.com/RishitToteja/Chext-X-ray-Images-Data-Set.git

2. Run the following commands:
   ```bash
   pip install pandas
   pip install numpy
   pip install tensorflow
