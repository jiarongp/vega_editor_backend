import os, sys
import argparse
import json
import torch
torch.manual_seed(42)
import numpy as np
from utils.utils import update_chart, load_json

from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.registry import Models

# Ax wrappers for BoTorch components
from ax.models.torch.botorch_modular.surrogate import Surrogate

# Experiment examination utilities
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.models.gp_regression import SingleTaskGP

# BoTorch components
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy


def optim_func() -> dict:
    """
    Optimisation function of BO.

    Args:
        bbox: bounding box coordinates

    Returns: score of the optimisation function
    """
    return {"loss_max": np.random.normal(0, 1, 1)[0]}

def bayesian_optim(chart_json: json, annotation:json, query: str, optim_path: str, chart_name:str):
    max_iter = 20
    gs = GenerationStrategy(
        steps=[
            GenerationStep(  # Initialization step
                # Which model to use for this step
                model=Models.SOBOL,
                # How many generator runs (each of which is then made a trial) to produce with this step
                num_trials=5,
                # How many trials generated from this step must be `COMPLETED` before the next one
                min_trials_observed=5,
            ),
            GenerationStep(  # BayesOpt step
                model=Models.BOTORCH_MODULAR,
                # No limit on how many generator runs will be produced
                num_trials=max_iter,
                model_kwargs={  # Kwargs to pass to `BoTorchModel.__init__`
                    "surrogate": Surrogate(SingleTaskGP),
                    "botorch_acqf_class": qLogNoisyExpectedImprovement,
                },
            ),
        ]
    )
    ax_client = AxClient(generation_strategy=gs)
    parameters = []
    for i in range(7):
        parameters.append(
            {
            "name": f"x{i}",
            "type": "range",
            "bounds": [0.0, 1.0],
            "value_type": "float",  # Optional, defaults to inference from type of "bounds".
            "log_scale": False,  # Optional, defaults to False.
        })
    # v_bar label rotation
    parameters.append({
        "name": f"x_rt",
        "type": "range",
        "bounds": [0.0, 2.0],
        "value_type": "int",  # Optional, defaults to inference from type of "bounds".
        "log_scale": False,  # Optional, defaults to False.
    })
    # bar orientation
    parameters.append({
        "name": f"x_ot",
        "type": "range",
        "bounds": [0.0, 1.0],
        "value_type": "int",  # Optional, defaults to inference from type of "bounds".
        "log_scale": False,  # Optional, defaults to False.
    })
    ax_client.create_experiment(
        name="baropt_experiment",
        parameters=parameters,    
        objectives={"loss_max": ObjectiveProperties(minimize=False)}
    )

    # Optimization loop
    for i in range(max_iter):
        parameterization, trial_index = ax_client.get_next_trial()
        update_chart(chart_json, parameterization, annotation)
        ax_client.complete_trial(trial_index=trial_index, raw_data=optim_func())

    best_parameters, values = ax_client.get_best_parameters()
    update_chart(chart_json, best_parameters, annotation, optim_path, chart_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/defaults/4488.json")
    parser.add_argument("--annot_path", type=str, default="./data/annotations/4488.json")
    parser.add_argument("--optim_path", type=str, default="./data/optims_randombaseline")
    args = vars(parser.parse_args())

    if '.json' in args['data_path']:
        chart_json, annot_json = load_json(args['data_path'], args['annot_path'])
        bayesian_optim(chart_json, annot_json, query=annot_json['tasks'][0]['question'], optim_path=args['optim_path'], chart_name=args['data_path'].split('/')[-1].strip('.json'))
    else: # batch processing
        for data_json in os.listdir(args['data_path']):
            chart_json, annot_json = load_json(os.path.join(args['data_path'], data_json), os.path.join(args['annot_path'], data_json))
            bayesian_optim(chart_json, annot_json, query=annot_json['tasks'][0]['question'], optim_path=args['optim_path'], chart_name=data_json.strip('.json'))
