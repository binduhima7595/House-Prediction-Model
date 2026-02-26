simple house price prediction MLOps project.
#dataset → preprocess → train model → track experiment → version data → API
Tools
******************
Python
MLflow → experiment tracking
DVC → data + pipeline
FastAPI → inference API

Step 0
******************
mkdir mlops-basic #create a folder
cd mlops-basic # go to folder
python -m venv venv # set environment
source venv/Scripts/activate # activate the source
pip install pandas scikit-learn mlflow dvc fastapi uvicorn joblib pyyaml # install requirements

STEP 1 — Create project structure
*******************
mlops-basic/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── api.py
│
├── models/
├── params.yaml
├── dvc.yaml
├── requirements.txt

STEP 2 — Add dataset
*******************
data/raw/housing.csv # i am using Kaggle data set

STEP 3 — Initialize Git + DVC
*******************
git init
dvc init
dvc add data/raw/housing.csv
git add .
git commit -m "init project"

Dataset now versioned

STEP 4 — Write preprocess.py
*******************
Goal: clean dataset → save processed

STEP 5 — Write params.yaml
*******************

STEP 6 — Write train.py (MLflow integration)
*******************

STEP 7 — Create DVC pipeline
*******************
dvc stage add -n preprocess -d src/preprocess.py -d data/raw/housing.csv -o data/processed/train.csv python src/preprocess.py
dvc stage add -n train -d src/train.py -d data/processed/train.csv -o models/model.pkl python src/train.py

STEP 8 — Run pipeline
*******************
dvc repro # runs the dvc.yaml

git add dvc.lock
git commit -m "successful pipeline run"

Verify : Check model exists
dir models
you can see model.pkl

Check MLflow experiment
mlflow ui
http://localhost:5000
check 
dvc repro
expected :
Stage 'preprocess' didn't change, skipping
Stage 'train' didn't change, skipping

What you have achieved
Git → code version
DVC → data + pipeline version
MLflow → experiment tracking
Pipeline reproducibility
if u want to change the code in .csv file, after code changes please run below command
git rm -r --cached data/processed/train.csv # we shall remove the train.csv first
git commit -m "stop tracking processed data with git"

dvc repro
git add .
git commit -m "update experiment"
git pull --rebase
git push


