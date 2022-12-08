# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Tessa van der Heiden created the model. It is logistic regression using the default hyperparameters in scikit-learn 0.24.2.

## Intended Use
This model should be used to predict the salary of US citizens based off a handful of attributes.

## Training Data
Data was obtained from: https://archive.ics.uci.edu/ml/datasets/census+income

## Evaluation Data

## Metrics
The model was evaluated using F1 score. The value is 0.60.
Additionally, per categorical category we obtained following interesting results:
Within "workclass" the model was performing well on "Without pay" and bad on "Self-employed".
It had a high score on "education" for category "Preschool", but low on "9th".
For "marital-status" the model did wel on "married" but not on "seperated".

## Ethical Considerations
No comment.

## Caveats and Recommendations
Much of these results can be explained by skewness: 
There weren't many values for categories in which the model performed badly. 
Example: there were 2x more men than women.
So one might want to process the data to remove skewness (e.g. up or down sampling).