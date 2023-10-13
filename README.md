# LinearRegression
1. LinearRegression.py is an independent model. It is the finalized model. It is able to process multi-feature datasets and finds optimized slope parameters w and bia b. And it able to predict any new sample.

2. LinearReg_1.py is a model similar to LinearRegression.py, but only processes a single feature dataset, but it can plot the graph since it is 2D. 

3. FeatureEngineering.py please discard. is the model I want to test how to increase complexity. How to perform f_wb = w1x^n + w2x^n-1 + w3x^n-2 ... + w(n+1)x+ b instead of f_wb = wx+b

4. LR_dataset2.py to test on a different dataset. I want to make sure the model works on any kind of digital dataset.

5. AppleAnalysis.py is an independent model. It predicts future prices of Apple stock(NOT A FINANCIAL MODEL)

6. FeatureScaling.py is the package. It helps preprocess the dataset. Scale training data and normalize them to avoid overflow. All of my models are required to import this package.

7. Simple model, no cross-validation, no parameter regularization (prone to overfit), no neuro also
