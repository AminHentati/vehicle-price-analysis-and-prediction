# Vehicle Price Analysis and Prediction

This project aims to analyze and predict vehicle prices using regression models and clustering techniques. By leveraging vehicle features such as model, year, transmission type, fuel type, and engine size, the objective is to predict the price of a vehicle with high accuracy. Additionally, the project segments vehicles into homogeneous clusters using Expectation-Maximization (EM) to provide insights into different price groups.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Workflow](#project-workflow)
- [Models Used](#models-used)
- [Evaluation Metrics](#evaluation-metrics)
- [Clustering](#clustering)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The core objectives of the project are:
- Predict vehicle prices using various regression models (Linear Regression, Support Vector Regression, Neural Networks).
- Cluster vehicles into homogeneous segments using the Expectation-Maximization algorithm.
- Handle preprocessing steps like treating missing values, encoding categorical variables, and scaling numerical variables.

## Dataset
The dataset used contains information on Ford vehicle prices with features such as:
- `model`: Vehicle model (e.g., Fiesta, Focus)
- `year`: Year of manufacture
- `transmission`: Transmission type (Automatic, Manual)
- `mileage`: Total miles driven
- `fuelType`: Fuel type (Petrol, Diesel, etc.)
- `tax`: Annual road tax in GBP
- `mpg`: Miles per gallon (fuel efficiency)
- `engineSize`: Size of the engine in liters
- `price`: Target variable (vehicle price)

The dataset is saved as `ford_Price.csv`.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vehicle-price-analysis.git
cd vehicle-price-analysis
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Project Workflow

### Data Loading and Exploration:
- Load the dataset and explore its structure, including the types of features (numerical and categorical) and any missing data.

### Data Preprocessing:
- Handle missing values.
- Convert categorical variables (like fuelType, transmission) into dummy/one-hot encoding.
- Scale numerical features (year, mileage, engineSize, etc.).

### Regression Models:
- Linear Regression: Simple, interpretable model.
- Support Vector Regression (SVR): Applied with different kernels (linear, polynomial, RBF, sigmoid) to capture non-linear relationships.
- Neural Networks (Keras): Built a feedforward neural network using the Sequential API from Keras for more complex relationships.

### Clustering (EM Algorithm):
- Use the Expectation-Maximization algorithm (Gaussian Mixture Model) to group the vehicles into clusters based on features such as year, engine size, and price.

### Model Evaluation:
- Use regression metrics like R² and RMSE to evaluate the performance of each model.

## Models Used
- Linear Regression: Basic regression model that fits a straight line to the data.
- Support Vector Regression (SVR):
  - Kernels used: linear, poly, rbf, sigmoid
- Neural Network (Keras Sequential model):
  - Hidden layers: Dense layer with ReLU activation
  - Output layer: Single node for regression

## Evaluation Metrics
Two evaluation metrics were used for regression model performance:
- R² (Coefficient of Determination): Measures how well the predicted values fit the true values. Ranges from 0 to 1, where 1 indicates perfect prediction.
- RMSE (Root Mean Squared Error): Measures the average deviation between predicted and actual values. Lower RMSE indicates better model performance.

## Clustering
The Expectation-Maximization (EM) algorithm was used to create homogeneous clusters based on vehicle features. Each cluster is analyzed to understand different pricing segments.

## Results

### Regression Model Performance:
- Linear Regression:
  - R²: 0.846
  - RMSE: 1858.79
- SVR (RBF Kernel):
  - R²: 0.095
  - RMSE: 4216.87
- Neural Network:
  - R²: 0.860
  - RMSE: 1780.65

### Clustering:
Two clusters were identified with distinct price ranges:
- Cluster 0: Vehicles with lower prices and older models.
- Cluster 1: Vehicles with higher prices and newer models.

## Technologies Used
- Python 3.8
- Pandas: Data manipulation and analysis
- NumPy: Numerical computing
- Scikit-learn: Machine learning models (Linear Regression, SVR)
- Keras: Neural Networks
- Matplotlib & Seaborn: Data visualization
- Gaussian Mixture Model (GMM): Clustering

## Contributing
Contributions are welcome! If you'd like to improve the project, feel free to fork the repository and submit a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
