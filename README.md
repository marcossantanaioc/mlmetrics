# mlmetrics
> A collection of metrics suitable to validate machine learning models


## Install from source

1.  `git clone https://github.com/marcossantanaioc/mlmetrics.git`

2.  `cd mlmetrics`

3.  `pip install -e .`

## How to use

**Regression**

```python
X_multi_reg, y_multi_reg = make_regression(n_samples=200, n_features=20, n_targets=5)
```

```python
model = RandomForestRegressor()

Xtrain,Xtest,ytrain,ytest = train_test_split(X_multi_reg, y_multi_reg,test_size=0.2)

model.fit(Xtrain, ytrain)

preds = model.predict(Xtest)

print(R2Score()(ytest, preds), MAEScore()(ytest, preds), MSEScore()(ytest, preds))
```

**Classification**

```python
X_class, y_class = make_classification(n_samples=200, n_features=20, n_classes=2)
```

```python
model = RandomForestClassifier()

Xtrain,Xtest,ytrain,ytest = train_test_split(X_class, y_class, stratify=y_class, test_size=0.2)

model.fit(Xtrain, ytrain)

preds = model.predict(Xtest)

print(RecallScore()(ytest, preds), PrecisionScore()(ytest, preds), MatthewsCorrCoef()(ytest, preds))
```
