# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Tessa van der Heiden created the model. It is a xgboost using the default hyperparameters in xgboost:
https://github.com/dmlc/xgboost/blob/master/doc/parameter.rst

## Intended Use
This model should be used to predict the salary of US citizens based off a handful of attributes.

## Training Data
Data was obtained from: https://archive.ics.uci.edu/ml/datasets/census+income

## Evaluation Data
20% of the data was used for evaluation. 

## Metrics
The model was evaluated using F1 score. The value is 0.78.
Additionally, per category we obtained following interesting results:
It had a low score on "education" for category "1st-4th", but high on others.
For "relationship" the model did well on "Wife" but not on "Own-child".

## Ethical Considerations
No comment.

## Caveats and Recommendations
Much of these results can be explained by skewness: 
There weren't many values for categories in which the model performed badly.
So one might want to process the data to remove skewness (e.g. up or down sampling).