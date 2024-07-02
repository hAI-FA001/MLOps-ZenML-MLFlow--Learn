### MLOps Using ZenML and MLFlow
Note: **This won't work on Windows** <br>
Specifically, the deployment and inference pipelines won't work. <br>
You will receive a "Daemon functionality is currently not supported on Windows" error. I've used an Ubuntu VM to circumvent this. <br>
You can still run the training pipeline using `python run_pipeline.py` and it will show up in the ZenML dashboard.
<br><br>
How to run:
- Put <a href="https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce">this</a> dataset (all its csv files) under "data"
- Run merge_csv.py as in `python merge_csv.py`. This should create a "olist_merged.csv" in "data"
- You can either:
  - Run the streamlit app using `streamlit run streamlit_app.py`
  - Run the deployment and inference pipelines using `python run_deployment.py --config deploy_and_predict --data-path "./data/olist_merged.csv"`

<br> <br>
ZenML commands (for reference): <br>
`zenml init` <br>
`zenml integration install sklearn mlflow -y` <br>
`zenml up` <br>
`zenml experiment-tracker register my_tracker_name --flavor=mlflow` <br>
`zenml model-deployer register my_deployer_name --flavor=mlflow` <br>
`zenml stack register my_stack_name -a default -o default -d my_deployer_name -e my_tracker_name --set` <br>
`zenml stack describe` <br>
`mlflow ui --backend-store-uri "path/to/file"` <br>
You can get this URI using `get_tracking_uri()` in `zenml.integrations.mlflow.mlflow_utils` <br>
