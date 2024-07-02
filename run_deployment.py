from pipelines.deployment_pipeline import (
    continuous_deployment_pipeline,
    inference_pipeline
)

import click

from rich import print
from typing import cast
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService


DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"

@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,
    help="Optionally you can choose to only run the deployment pipeline to train and deploy a model ('deploy'), or to only run a prediction pipeline against the"
    "deployed model ('predict'). By default, both will be run ('deploy_and_predict')"
)
@click.option(
    "--min-accuracy",
    default=0.92,
    help="Minimum accuracy required to deploy the model"
)
@click.option(
    "--data-path",
    help="Path to CSV file containing your data"
)
def run_deployment(config: str, min_accuracy: float, data_path: str):
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT

    if deploy:
        print("\n\n\n", data_path, min_accuracy)
        continuous_deployment_pipeline(
            data_path=data_path,
            min_accuracy=min_accuracy,
            workers=3,
            timeout=60
            )
    if predict:
        inference_pipeline(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
        )

    print(
        f"""You can run:
[italic green]          mlflow ui --backend-store-uri "{get_tracking_uri()}" [/italic green]
...to inspect your experiment runs within the MLFlow UI.
You can find your runs tracked within the `mlflow_example_pipeline` experiment. There, you'll also be able to compare two or more runs.

        """
    )

    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        model_name="model",
    )


    if existing_services:
        service = cast(MLFlowDeploymentService, existing_services[0])
        service.start(timeout=60)
        if service.is_running:
            print(f"""The MLFlow prediction server is running locally as a daemon process service and accepts inference requests at:
                  {service.get_prediction_url()}
To stop the service, run
[italic green] `zenml model-deployer models delete {str(service.uuid)}`[/italic green]
""")
        elif service.is_failed:
            print(f"""The MLFlow prediction server is in a failed state:
Last state: `{service.status.state.value}`
Last error: `{service.status.last_error}`
""")
    else:
        print("""No MLFlow prediction server is running. The deployment pipeline must run first to train a model and deploy it.
Execute the same command with `--deploy` argument to deploy a model.
""")
    

if __name__ == "__main__":
    run_deployment()

