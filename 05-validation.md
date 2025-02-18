---
title: Validation
teaching: 20
exercises: 10
---

::::::::::::::::::::::::::::::::::::::: objectives

- Train a model to predict patient outcomes on a held-out test set.
- Use cross validation as part of our model training process.

::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::: questions

- What is meant by model accuracy?
- What is the purpose of a validation set?
- What are two types of cross validation?
- What is overfitting?

::::::::::::::::::::::::::::::::::::::::::::::::::

## Accuracy

One measure of the performance of a classification model is accuracy. Accuracy is defined as the overall proportion of correct predictions. If, for example, we take 50 shots and 40 of them hit the target, then our accuracy is 0.8 (40/50).

![](fig/japan_ren_hayakawa.jpg){alt='Ren Hayakawa Archery Olympics' width="600px"}

Accuracy can therefore be defined by the formula below:

$$ Accuracy = \frac{Correct\ predictions}{All\ predictions}$$

What is the accuracy of our model at predicting in-hospital mortality?

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Convert outcome to categorical type
categories = ['ALIVE', 'EXPIRED']
cohort['actualhospitalmortality'] = pd.Categorical(cohort['actualhospitalmortality'], categories=categories)

# Encode categorical values
cohort['actualhospitalmortality_enc'] = cohort['actualhospitalmortality'].cat.codes
cohort[['actualhospitalmortality_enc','actualhospitalmortality']].head()


# Define features and outcome
features = ['apachescore']
outcome = ['actualhospitalmortality_enc']

# Partition data into training and test sets
X = cohort[features]
y = cohort[outcome]
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

# Restructure data for model input
x_train = x_train.values.reshape(-1, 1)
y_train = y_train.values.ravel()
x_test = x_test.values.reshape(-1, 1)
y_test = y_test.values.ravel()

# Train Logistic Regression model
logreg = LogisticRegression(random_state=0)
logreg.fit(x_train, y_train)

# Train Decision Tree model
tree = DecisionTreeClassifier(random_state=0)
tree.fit(x_train, y_train)

# Generate predictions
y_hat_train_logreg = logreg.predict(x_train)
y_hat_test_logreg = logreg.predict(x_test)

y_hat_train_tree = tree.predict(x_train)
y_hat_test_tree = tree.predict(x_test)

# Accuracy on training set
acc_train_logreg = np.mean(y_hat_train_logreg == y_train)
acc_train_tree = np.mean(y_hat_train_tree == y_train)

print(f'Logistic Regression - Accuracy on training set: {acc_train_logreg:.2f}')
print(f'Decision Tree - Accuracy on training set: {acc_train_tree:.2f}')

# Accuracy on test set
acc_test_logreg = np.mean(y_hat_test_logreg == y_test)
acc_test_tree = np.mean(y_hat_test_tree == y_test)

print(f'Logistic Regression - Accuracy on test set: {acc_test_logreg:.2f}')
print(f'Decision Tree - Accuracy on test set: {acc_test_tree:.2f}')
```

```output
TO-DO
Accuracy on training set: 0.86
Accuracy on test set: 0.82
```

Not bad! There was a slight drop in performance on our test set, but that is to be expected.

## Validation set

Machine learning is iterative by nature. We want to improve our model, tuning and evaluating as we go. This leads us to a problem. Using our test set to iteratively improve our model would be cheating. It is supposed to be "held out", not used for training! So what do we do?

The answer is that we typically partition off part of our training set to use for validation. The "validation set" can be used to iteratively improve our model, allowing us to save our test set for the \*final\* evaluation.

![](fig/training_val_set.png){alt='Validation set' width="600px"}

## Cross validation

Why stop at one validation set? With sampling, we can create many training sets and many validation sets, each slightly different. We can then average our findings over the partitions to give an estimate of the model's predictive performance

The family of resampling methods used for this is known as "cross validation". It turns out that one major benefit to cross validation is that it helps us to build more robust models.

If we train our model on a single set of data, the model may learn rules that are overly specific (e.g. "all patients aged 63 years survive"). These rules will not generalise well to unseen data. When this happens, we say our model is "overfitted".

If we train on multiple, subtly-different versions of the data, we can identify rules that are likely to generalise better outside out training set, helping to avoid overfitting.

Two popular of the most popular cross-validation methods:

- K-fold cross validation
- Leave-one-out cross validation

## K-fold cross validation

In K-fold cross validation, "K" indicates the number of times we split our data into training/validation sets. With 5-fold cross validation, for example, we create 5 separate training/validation sets.

![](fig/k_fold_cross_val.png){alt='5-fold validation' width="600px"}

With K-fold cross validation, we select our model to evaluate and then:

1. Partition the training data into a training set and a validation set. An 80%, 20% split is common.
2. Fit the model to the training set and make a record of the optimal parameters.
3. Evaluate performance on the validation set.
4. Repeat the process 5 times, then average the parameter and performance values.

When creating our training and test sets, we needed to be careful to avoid data leaks. The same applies when creating training and validation sets. We can use a `pipeline` object to help manage this issue.

```python
from numpy import mean, std
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Define dataset
X = x_train
y = y_train

# Define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

# Logistic Regression pipeline
logreg_pipeline = Pipeline([
    ('scaler', MinMaxScaler()), 
    ('model', LogisticRegression())
])

# Decision Tree pipeline
tree_pipeline = Pipeline([
    ('model', DecisionTreeClassifier())
])

# Evaluate Logistic Regression
logreg_scores = cross_val_score(logreg_pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Logistic Regression - Cross-validation accuracy: mean=%.2f%%, std=%.2f%%' % (mean(logreg_scores)*100, std(logreg_scores)*100))

# Evaluate Decision Tree
tree_scores = cross_val_score(tree_pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Decision Tree - Cross-validation accuracy: mean=%.2f%%, std=%.2f%%' % (mean(tree_scores)*100, std(tree_scores)*100))
```

```output
Cross-validation accuracy, mean (std): 81.53 (3.31)
```

Leave-one-out cross validation is the same idea, except that we have many more folds. In fact, we have one fold for each data point. Each fold we leave out one data point for validation and use all of the other points for training.



:::::::::::::::::::::::::::::::::::::::: keypoints

- Validation sets are used during model development, allowing models to be tested prior to testing on a held-out set.
- Cross-validation is a resampling technique that creates multiple validation sets.
- Cross-validation can help to avoid overfitting.

::::::::::::::::::::::::::::::::::::::::::::::::::


