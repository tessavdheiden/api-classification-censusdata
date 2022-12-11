Code for an API that allows classifying a sample from the census dataset.

# Environment Set up
* Download and install conda if you don’t have it already.
* Create a new environment with conda:
```bash
> conda create -n [envname] "python=3.8" scikit-learn dvc pandas numpy pytest jupyter jupyterlab fastapi uvicorn -c conda-forge
```

## GitHub Actions
* Setup GitHub Actions on your repository. You can use one of the pre-made GitHub Actions if at a minimum it runs pytest and flake8 on push and requires both to pass without error.
* Use autopep to reformat your code:
```bash
> autopep8 --in-place --aggressive --aggressive [script-name].py
```
* Use pytest before committing to make sure tests run properly:
```bash
> pytest
```
* Commit and push regularly:
```bash
> git commit -m -a "[message]"
> git push 
```

## Data
* Information on the dataset can be found <a href="https://archive.ics.uci.edu/ml/datasets/census+income" target="_blank">here</a>.

## Model
* Code contains a [machine learning model](https://github.com/dmlc/xgboost/blob/master/doc/parameter.rst) that trains on the clean data and saves the model:
```bash
> python starter/train_model.py
```
* Code for evaluation can be executed by:
```bash
> python starter/eval_model.py
```
* See model_card_template.md for the model card.

## API 

* The code contains a RESTful API using FastAPI:
   * GET on the root giving a welcome message.
   * POST that does model inference. Try it out yourself by:

## API Deployment

* Create a free Heroku account (for the next steps you can either use the web GUI or download the Heroku CLI).
* Connecting to Heroku, create an app with a specific name and to specify it as Python app:
```bash
> heroku create [name-of-app] --buildpack heroku/python
```
* Create another remote on Heroku: 
```bash
> heroku git:remote --app [name-of-app]
```
* Optionally, check if the remote is there:
```bash
> git remote -v
```
* Lauch the app:
```bash
> git push heroku main
```
* Now we can check out the app at https://[name-of-app].herokuapp.com/
* There's a script that uses the requests module to do one POST on your live API:
```bash
> python sample_request.py
```
