# Machine-Learning-Deep-Learning
Team task at Startup Campus (Artificial Intelligence)

### Case 1: Machine Learning (Supervised: Classification)
**Predicting Customer Churn with Classification Models**  
The objective of this case is to predict customer churn status in banking using classification models. We compare three different models to identify the most accurate one, applying hyperparameter tuning and feature selection to optimize performance.
#### Steps

1. **Data Preprocessing**:
   - Remove irrelevant columns.
   - Apply one-hot encoding to categorical features (Geography and Gender).
   - Scale the features using MinMaxScaler.

2. **Train-Test Split**:
   - Split data into training and testing sets (75% train, 25% test).

3. **Model Training and Evaluation**:
   - **Logistic Regression**
   - **Random Forest Classifier**
   - **Gradient Boosting Classifier**

   Each model is trained and tuned using `GridSearchCV` to find optimal hyperparameters. Evaluation metrics include accuracy, precision, recall, F1 score, and confusion matrix.

4. **Conclusion**: Identify the model with the best performance based on evaluation metrics, with an emphasis on accuracy and balanced detection of churned customers.

---

### Case 2: Data Segmentation with KMeans Clustering

#### Overview
In this case, you will perform data segmentation using the KMeans clustering algorithm. The objective is to determine the optimal number of clusters for the dataset and visualize the resulting segmentation.

#### Dataset
- **Content**: The dataset contains two columns (`x` and `y`), representing data points that need to be clustered.

#### Objective
1. **Optimal Clusters**: Find the optimal number of clusters by calculating the silhouette score for different values of `k` (clusters).
2. **Clustering**: Apply the KMeans algorithm using the optimal number of clusters.
3. **Data Labeling and Visualization**: Add the cluster labels to the dataset and visualize the clusters using `seaborn`.

#### Tasks
1. Perform clustering with various values of `k` and calculate silhouette scores to determine the best number of clusters.
2. Implement KMeans clustering with the chosen number of clusters.
3. Add cluster labels to the dataset and visualize the segmentation using `seaborn`.

#### Expected Outcome
You will generate a clustering model based on KMeans, visualize the results, and evaluate the clustering performance with silhouette scores.

---

### Case 3: House Price Prediction using MLP (Multilayer Perceptron)

#### Overview
This project aims to build a house price prediction model using the California Housing dataset and apply deep learning techniques with TensorFlow-Keras. The model used is a Multilayer Perceptron (MLP) to predict house prices based on various features available in the dataset.

#### Objectives
- Build a deep learning model using MLP to predict house prices.
- Perform data preprocessing to prepare the dataset for training.
- Evaluate the model's performance using error metrics like MSE (Mean Squared Error).

#### Analysis & Modelling Tools
- **TensorFlow** & **Keras**: For building and training the deep learning model.
- **Pandas**: For data manipulation and preprocessing.
- **Scikit-learn**: For splitting the data into training, validation, and test sets.
- **Matplotlib**: For visualizing results and model evaluation.

---

### Case 4: Credit Fraud Detection using Deep Learning

#### Overview
This project focuses on detecting fraudulent credit card transactions using a Multilayer Perceptron (MLP) neural network model. The objective is to classify transactions as either legitimate or fraudulent based on various transaction features. The model will be trained on a labeled dataset, and the goal is to achieve high classification accuracy.

#### Dataset:
The dataset used in this project is the **Credit Card Fraud 2023** dataset. This dataset is a collection of anonymized credit card transactions. 

#### Objectives
- **Classify fraudulent transactions**: Build and train an MLP model to classify transactions as fraudulent or legitimate.
- **Improve model accuracy**: Tune the architecture and hyperparameters of the model to achieve the highest possible accuracy and generalization performance.
- **Evaluate performance**: Use metrics such as accuracy, precision, recall, and F1-score to evaluate the model’s performance, especially considering the class imbalance.

#### Analysis & Modelling Tools
- **Python**: The programming language used for data preprocessing, model training, and evaluation.
- **TensorFlow/Keras**: Deep learning framework for constructing and training the MLP model.
- **Pandas & NumPy**: Libraries for data manipulation, cleaning, and analysis.
- **Matplotlib & Seaborn**: Visualization tools for exploratory data analysis and result presentation.
- **Scikit-learn**: For metrics and data splitting (train-test) and handling class imbalance.

---
