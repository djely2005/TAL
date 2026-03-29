# Model Submissions


## Project Overview

The task 1 involves training multiple classification models on a labeled corpus of speeches to predict which president spoke each sentence in the test dataset. The dataset shows significant class imbalance (85% Chirac, 15% Mitterrand).

The task 2 ...


---

## Submission Files & Models 

### **submission-pres-1.csv** - Baseline Logistic Regression

**Model**: Logistic Regression with balanced class weights

**Configuration**:
- Algorithm: Logistic Regression (`LogisticRegression`)
- Regularization parameter C: Automatically adjusted based on validation performance
- Class weight: balanced
- Data balancing: Under-sampling on training set (reducing majority class)
- Post-processing: Gaussian smoothing applied to predictions

**Description**:
This is the baseline model that establishes initial performance metrics. It uses a balanced class weight to penalize misclassifications of the minority class (Mitterrand). Under-sampling reduces the training set by randomly selecting equal numbers of both classes, which helps the model learn Mitterrand's distinctive vocabulary while preventing lazy predictions.

---

### **submission-pres-3.csv** - Logistic Regression with Over-Sampling

**Model**: Logistic Regression with over-sampling for class balance

**Configuration**:
- Algorithm: Logistic Regression
- Regularization parameter C: Automatically adjusted based on validation performance
- Class weight: balanced
- Data balancing: Over-sampling on training set (duplicating minority class with replacement)
- Post-processing: Gaussian smoothing applied to predictions

**Description**:
This model improves upon the baseline by using **over-sampling** instead of under-sampling. Rather than discarding majority class samples, it duplicates minority class samples (Mitterrand) with replacement to reach equal class distribution. This approach preserves all available training data and typically yields better performance than under-sampling when the original dataset size is limited. The model benefits from more complete vocabulary coverage of both speakers.

---

### **submission-pres-5.csv** - Logistic Regression with Adaptive Class Weights

**Model**: Logistic Regression with class_weight='balanced'

**Configuration**:
- Algorithm: Logistic Regression
- Regularization parameter C: Automatically adjusted based on validation performance
- Class weight: 'balanced' (weights automatically computed proportional to inverse class frequency)
- Data balancing: No explicit sampling - class imbalance handled via learned weight adjustments
- Post-processing: Gaussian smoothing applied to predictions (kernel size: 8)

**Description**:
This approach represents a purely algorithmic solution to class imbalance. Rather than resampling the training data, scikit-learn's `class_weight='balanced'` feature automatically computes inverse class frequency weights during training. This means misclassifying the minority class (Mitterrand) carries a higher penalty in the loss function. This elegant solution avoids data manipulation while maintaining the original dataset structure and is often more computationally efficient than resampling methods.

**Post-Processing Enhancement**:
All submissions employ **Gaussian smoothing** on predicted probabilities to exploit the temporal structure of speeches. The underlying principle is that consecutive sentences from the same speaker are likely clustered together. A Gaussian kernel is convolved across predictions to smooth out isolated anomalies, improving overall accuracy.

---

### **submission-pres-2.csv** - Support Vector Machine (SVM)

**Model**: Support Vector Machine with probability calibration

**Configuration**:
- Algorithm: LinearSVC with CalibratedClassifierCV
- Regularization parameter C: Automatically adjusted based on validation performance
- Class weight: balanced
- Calibration: 5-fold cross-validation with sigmoid method
- Vectorization: TF-IDF with max_features=10,000
- Post-processing: Gaussian smoothing applied to predictions

**Description**:
This model represents a different algorithmic approach compared to the logistic regression variants. **Support Vector Machines** find an optimal separating hyperplane in the feature space that maximizes the margin between classes. LinearSVC (linear kernel) is computationally efficient and works well with the sparse TF-IDF feature vectors.

The model includes `CalibratedClassifierCV` to convert raw SVM decision scores into calibrated probability estimates (0-1 range), which is necessary for generating probability-based predictions. The calibration uses 5-fold cross-validation in a sigmoid method to map decision values to well-calibrated probabilities.

**Advantages over Logistic Regression**:
- Larger margin provides better generalization
- Effective in high-dimensional sparse spaces (typical for text data)
- Different decision boundary complexity that may capture non-linear patterns

---

## Common Techniques Across All Models

### TF-IDF Vectorization
- **N-grams (1,2)**: Captures unigrams and bigrams (e.g., "française" + "politique étrangère")
- **min_df=2**: Ignores words appearing only once (likely typos or noise)
- **max_features**: Limited to most frequent features for computational efficiency
- **Effect**: Distinguishes speaker-specific vocabulary while penalizing overly common generic political terms

### Gaussian Smoothing
Applies temporal coherence to predictions via convolution with a Gaussian kernel. The intuition is that speeches are typically continuous - isolated contradicting predictions are anomalies that should be smoothed out using context from neighboring sentences.

### Class Imbalance Mitigation
Three different strategies implemented:
1. **Under-sampling**: Reduces majority class for balanced dataset
2. **Over-sampling**: Duplicates minority class for balanced dataset  
3. **Class weights**: Algorithmic weighting during model training

---

## Model Performance Comparison

Each model was optimized using validation set metrics, primarily focusing on:
- **F1-Score for Mitterrand** (minority class) - primary optimization target
- **AUC (ROC)** - probability quality metric
- **Recall for Mitterrand** - ensures we identify minority class instances
- **Accuracy** - overall correctness

The different approaches represent trade-offs between:
- Data preservation vs. data modification
- Algorithmic simplicity vs. geometric margin maximization
- Computational efficiency vs. predictive power

---

## Project Dataset

- **Train/Validation**: 
`Dataset/corpus.tache1.learn.utf8` (~58,000 sentences)
  - Class distribution: ~85% Chirac, ~15% Mitterrand
`movies1000.zip`
  
- **Test**: 
`Dataset_test/corpus.tache1.test.utf8` (~8,000 sentences)
  - Used for generating final predictions in submission files

`testSentiment.txt`
