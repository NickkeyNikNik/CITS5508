# CITS5508
 
# CITS5508 - Assignment 1: Logistic Regression and k-NN Classification

This project was completed as part of the CITS5508 course and focuses on building and evaluating machine learning models to classify images of sandals and sneakers. The two primary algorithms explored in this project are Logistic Regression (LR) and k-Nearest Neighbors (k-NN), both of which are used to distinguish between these two classes. Key techniques such as data scaling, model tuning, and performance evaluation are covered in this assignment.

## Project Overview

The project involves using logistic regression and k-NN to classify images from a dataset containing sandals and sneakers. Various aspects of machine learning are explored, such as:
- **Data Preprocessing**: Scaling the image pixel values to improve performance.
- **Model Tuning**: Finding the optimal hyperparameters (e.g., learning rates for logistic regression, k-values for k-NN).
- **Evaluation Metrics**: Using precision, recall, false positive rates, and misclassification rates to evaluate model performance.
- **Cross-Validation**: Applied to ensure models generalize well on unseen data.

### Key Features:
1. **Data Preprocessing with Standard Scaling**: The pixel values in the images range from 0 to 255. Using the `StandardScalar()` function, the data was scaled to improve computational speed and accuracy, particularly for algorithms sensitive to distance measures like k-NN.
   
2. **Instance Splitting**: The dataset of 13,988 instances was split into 11,988 training instances and 2,000 test instances, ensuring a balanced distribution of sandals and sneakers across the sets.

3. **Model Training and Tuning**:
   - Logistic Regression: Tuning of the learning rate and regularization parameter `C` was performed. Learning rates between 0.0001 and 0.0005 were compared to identify the best trade-off between convergence speed and accuracy.
   - k-NN: Different values for `k` (the number of neighbors) were tested to identify the optimal model for classification.

4. **Model Evaluation**: Various evaluation metrics were used, including precision, recall, and false positive rates. Precision-recall curves and confusion matrices were generated to assess the models' performance.
   - Logistic regression models were evaluated based on their precision, recall, and misclassification rates.
   - k-NN models were compared based on their ability to generalize, using precision and recall to judge their performance.

### Key Findings:
- **Optimal Logistic Regression Parameters**: The optimal learning rate was found to be 0.0001 with a regularization parameter `C` of 10^-2. These values struck a balance between low misclassification rates and good generalization across both the training and validation sets.
- **k-NN Performance**: The k-NN model with `k=6` provided the best performance on the test set, minimizing the test misclassification rate. The precision of the k-NN model was very high, but it had slightly lower recall compared to logistic regression models LR3 and LR4.

## Results

### Logistic Regression:
- **Optimal C Value**: The best regularization parameter was determined to be `C = 10^-2`, with the model achieving low misclassification rates and good generalization.
- **Precision-Recall Tradeoff**: A precision-recall curve was generated to assess the model's performance in various scenarios. The optimal threshold was found to be 0.72, with an F1 score of 0.95.
- **Confusion Matrix**: Detailed analysis of the false positives and false negatives was conducted, with insights provided into which features the model found challenging to classify.

### k-NN Model:
- **Best k Value**: The best value for `k` was found to be `k=6`, which minimized the test misclassification rate while balancing precision and recall.
- **Performance Metrics**: The k-NN model achieved a precision of 0.992 and a recall of 0.905, with a false positive rate of 0.008. Despite its high precision, it had slightly lower recall than some logistic regression models.

## Conclusion

This project demonstrated the successful application of both logistic regression and k-NN for image classification tasks. By tuning hyperparameters and carefully evaluating performance using cross-validation, models were developed that effectively distinguished between sandals and sneakers with high precision and recall. The use of scaling, regularization, and validation techniques ensured that the models were both efficient and accurate in their predictions.

## Improvements

Several areas for improvement were identified for future work:
- **Increasing Dataset Size**: Collecting more data to improve model generalization.
- **Ensemble Methods**: Using ensemble techniques such as bagging or boosting to further enhance performance.
- **Advanced Techniques**: Exploring deep learning models for potentially better image classification results.

# CITS5508 - Assignment 2: Decision Tree and Random Forest Classifiers for Breast Cancer Diagnosis

This project, completed as part of the CITS5508 course, focuses on building, evaluating, and fine-tuning machine learning models using decision trees and random forests to classify breast cancer tumors as malignant or benign. The assignment covers various stages of data preprocessing, model training, hyperparameter tuning, and performance evaluation, ensuring a robust comparison of models.

## Project Overview

The objective of this project is to predict whether a breast tumor is malignant or benign using classification algorithms. The models used include Decision Trees and Random Forests, with the goal of determining the best model based on accuracy, precision, and recall. Various techniques such as grid-search, cross-validation, and feature importance analysis are explored.

### Key Components:
1. **Data Preprocessing**: 
   - Reordering columns and visualizing relationships between features to identify potential correlations.
   - Removing highly correlated features (those with correlations higher than 0.97) to prevent multicollinearity.
   
2. **Decision Tree Classifier**:
   - A decision tree classifier was trained using 80% of the data, and its performance was evaluated on both training and test sets.
   - Hyperparameter tuning with grid-search and 10-fold cross-validation was applied to find the optimal values for `max_depth`, `min_samples_split`, and `min_samples_leaf`.
   
3. **Random Forest Classifier**:
   - A Random Forest model was built using grid-search to determine the best combination of `n_estimators` and `max_depth`.
   - Feature importance was analyzed to further refine the model.
   
4. **Performance Evaluation**:
   - Multiple metrics, including accuracy, precision, recall, and confusion matrices, were used to assess the performance of each model on different data splits.
   - The impact of training size on model performance was analyzed, with splits ranging from 50%-50% to 90%-10%.
   - Models were also compared based on their performance with reduced and complete feature sets.

## Key Results and Findings

### Decision Tree Classifier:
- **Initial Performance**: The decision tree model achieved 96% accuracy on the test set with an identical precision and recall of 0.96. However, the model showed signs of overfitting, as evidenced by its perfect scores on the training data.
- **Overfitting Insights**: A tree with 9 levels was built, which indicated potential overfitting, especially as some leaves contained very few samples. The model performed well on the training data but showed slight drops in generalization when tested on unseen data.
- **Hyperparameter Tuning**: Fine-tuning the hyperparameters (max depth, min samples split, min samples leaf) resulted in improved performance. The tuned model achieved 94% accuracy, precision, and recall on the validation set.

### Random Forest Classifier:
- **Improved Performance**: The Random Forest classifier outperformed the decision tree model, achieving a 98% accuracy, precision, and recall after tuning. The ensemble method reduced both type I and type II errors, making it a more robust and reliable model.
- **Feature Importance**: The analysis revealed that key features, such as 'mean smoothness' and 'worst concave points,' had the greatest influence on model performance, while others with less than 1% importance were dropped to simplify the model.

### Comparison Between Models:
- **Consistency Across Splits**: Both models were tested across multiple data splits, showing consistent performance with slight variations in precision and recall, depending on the specific configuration of the test set.
- **Training Size Impact**: The model's performance generally improved as the training set increased, with a 70%-30% split providing the best overall performance balance. At smaller splits (50%-50%), the models tended to underfit due to insufficient training data.

## Conclusion

This project demonstrated the effectiveness of decision trees and random forests in classifying breast cancer tumors. The Random Forest model, with its ensemble learning approach, proved to be more accurate and generalizable than the single decision tree. Through careful feature selection and hyperparameter tuning, both models achieved high accuracy and precision, with the Random Forest model excelling in minimizing misclassifications.

## Future Improvements
1. **Increase Dataset Size**: Collecting more data could further enhance model performance, particularly for capturing complex patterns in breast cancer diagnosis.
2. **Ensemble Techniques**: Combining decision trees with other ensemble methods like boosting could improve the model's ability to generalize.
3. **Complex Models**: Exploring more complex models, such as deep learning, could capture intricate features in the data that traditional methods may miss.

# CITS5508 - Assignment 3: Regression Models and Clustering Analysis for Housing Prices

This project, completed as part of the CITS5508 course, focuses on building and evaluating regression models to predict housing prices, alongside applying clustering techniques to understand patterns in housing data. The project uses models like linear regression, Lasso, Ridge, and Decision Tree regression, and compares their performance using RMSE. Additionally, it explores clustering techniques such as hierarchical clustering and k-means to uncover meaningful groupings in the data.

## Project Overview

The objective of this project is to predict housing prices based on a variety of features such as the number of rooms, bedrooms, and population in different districts. The models and techniques used include:
- **Linear Regression**: To model relationships between features and housing prices.
- **Lasso Regression**: A regularized regression model that helps manage feature selection and multicollinearity.
- **Ridge Regression**: Similar to Lasso, but with a different regularization approach that shrinks coefficients uniformly.
- **Decision Tree Regression**: Captures non-linear relationships between features for better predictive accuracy.
- **Clustering Analysis**: Explores housing data using hierarchical clustering and k-means to identify natural groupings.

### Key Components:
1. **Exploratory Data Analysis (EDA)**:
   - Visualized histograms of non-categorical features and computed the correlation matrix to assess relationships between variables.
   - Scatter plots were used to display the relationships between features and the target variable (housing prices).
   
2. **Regression Models**:
   - Trained Linear, Lasso, and Ridge Regression models using both original and standardized datasets.
   - Hyperparameter tuning using grid search and cross-validation was applied to find the best parameters for Lasso and Ridge Regression.
   - Decision Tree Regression was also trained, and its performance was compared against the linear models.
   
3. **Clustering Analysis**:
   - Hierarchical clustering and k-means were applied to discover natural groupings in the housing data.
   - Principal Component Analysis (PCA) was performed to reduce dimensionality and visualize the clusters.
   - Silhouette scores were used to determine the optimal number of clusters for k-means clustering.

## Key Results and Findings

### Regression Models:
- **Linear and Lasso Regression**:
   - Both models exhibited similar RMSE values, with the linear regression model proving to be relatively robust against feature scaling. The Lasso model, however, was more sensitive to the regularization parameter (α = 100), indicating potential underfitting.
   
- **Decision Tree Regression**:
   - The Decision Tree model outperformed both linear and regularized models in terms of RMSE on the test set. Its ability to capture non-linear relationships led to improved accuracy.
   
- **Feature Engineering**:
   - New features like `meanRooms`, `meanBedrooms`, and `meanOccupation` were created to better capture housing characteristics. These features slightly improved model performance after standardization.

### Clustering Analysis:
- **Hierarchical Clustering**:
   - Districts were grouped into four clusters based on features like the number of rooms and median income. These clusters reflected significant differences in geographic locations and housing characteristics.
   
- **K-Means Clustering**:
   - K-means clustering with k=4 was applied, resulting in similar clusters to the hierarchical method but with slight adjustments to the centroid positions. PCA combined with k-means revealed clusters that aligned well with geographic proximity.
   
- **PCA and Dimensionality Reduction**:
   - PCA was used to reduce the feature space, preserving 90% of the variance in just five components. The clustering based on PCA scores provided more balanced cluster sizes and helped interpret the underlying structure of the data.

### Optimal Models:
- **Lasso and Ridge Regression**:
   - Lasso and Ridge regression models benefited from regularization, managing multicollinearity, but struggled with capturing non-linear relationships in the data.
   
- **Decision Tree Regression**:
   - The best performing model in terms of RMSE was the Decision Tree regression, which captured non-linear patterns effectively and performed better than both Lasso and Ridge models.

## Conclusion

This project successfully demonstrates the application of various regression models and clustering techniques to predict housing prices. The Decision Tree model provided the best performance, indicating that non-linear relationships between features and housing prices play a significant role in improving predictive accuracy. Clustering analysis revealed meaningful groupings in the data, which could be leveraged to improve the models further. Feature engineering and dimensionality reduction techniques like PCA played a crucial role in optimizing the model's performance.
