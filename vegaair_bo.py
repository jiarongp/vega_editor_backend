import os, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import json
import torch
torch.manual_seed(42)
from PIL import Image
import numpy as np
from typing import List
from model.model import SalFormer
from transformers import AutoImageProcessor, AutoTokenizer, BertModel, SwinModel
# import cv2
from utils.utils import update_chart
from utils.text_ocr import txt_loss
from utils.visual_density import vd_loss
from utils.metrics import wave_metric

from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.registry import Models

# Ax wrappers for BoTorch components
from ax.models.torch.botorch_modular.surrogate import Surrogate

# Experiment examination utilities
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.models.gp_regression import SingleTaskGP

# BoTorch components
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy


def predict(ques: str) -> List:
    """
    Execute the prediction.

    Args:
        ques: a question string to feed into VisSalFormer

    Returns: [list]
        - heatmap from VisSalFormer (np.array)
        - Average WAVE score across pixels (float, [0, 1))
    """
    image = Image.open('data/chart.png').convert("RGB")
    img_pt = image_processor(image, return_tensors="pt").to(device)
    inputs = tokenizer(ques, return_tensors="pt").to(device)

    mask = model(img_pt['pixel_values'], inputs)
    mask = mask.detach().cpu().squeeze().numpy()
    heatmap = (mask * 255).astype(np.uint8)
    gary_image = image.convert('L')

    # image_np = np.array(image)
    # image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2RGBA)
    heatmap = np.resize(heatmap, (image.size[1], image.size[0]))
    return [heatmap, image, np.array(gary_image)]


def optim_func(predictions: List, bboxes: List[np.ndarray], chart_json: json) -> dict:
    """
    Optimisation function of BO.

    Args:
        predictions[list]: same as the output of predict()
        predictions[0]: heatmap from VisSalFormer (np.array)
        predictions[1]: original image (Image)
        predictions[2]: gray image (np.array)
        bbox: bounding box coordinates

    Returns: score of the optimisation function
    """
    # Text Loss is a metric that measures the readability of texts, ranges [0, 1]    
    TXT_OCR = txt_loss(predictions[1], chart_json) * 255.
    # WAVE is a metric that measures how close the colors in the image are to the preferred colors from human [0, 1]
    WAVE = wave_metric(predictions[1]) * 255.
    # heatmap_mean is the mean value of saliency maps in the bounding box (larger than 8, which thresholds the whitespaces out)
    heatmap_mean = 0
    for bbox in bboxes:
        bbox_heapmap = predictions[0][bbox[1]:bbox[3], bbox[0]:bbox[2]]
        if bbox_heapmap[bbox_heapmap>8].size > 0:
            heatmap_mean += np.mean(bbox_heapmap[bbox_heapmap>8]) # thresholding the low salient pixels, so that the size of bounding box won't matter that much
    if len(bboxes) == 0:
        return {"loss_max": (WAVE + TXT_OCR - 512 * vd_loss(predictions[2]), 0.0)}
    return {"loss_max": (WAVE + TXT_OCR + 4 * heatmap_mean / len(bboxes) - 512 * vd_loss(predictions[2]), 0.0)}

def bayesian_optim(chart_json: json, annotation:json, query: str, optim_path: str, chart_name:str):
    max_iter = 30
    gs = GenerationStrategy(
        steps=[
            GenerationStep(  # Initialization step
                # Which model to use for this step
                model=Models.SOBOL,
                # How many generator runs (each of which is then made a trial)
                # to produce with this step
                num_trials=5,
                # How many trials generated from this step must be `COMPLETED`
                # before the next one
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
    # best_iter = 0
    for i in range(max_iter):
        parameterization, trial_index = ax_client.get_next_trial()
        # Local evaluation here can be replaced with deployment to external system.
        bboxes = update_chart(chart_json, parameterization, annotation)
        predictions = predict(query)
        ax_client.complete_trial(trial_index=trial_index, raw_data=optim_func(predictions, bboxes, chart_json))

    best_parameters, values = ax_client.get_best_parameters()
    update_chart(chart_json, best_parameters, annotation, optim_path, chart_name)
    # print('best iter appears at', best_iter, best_parameters, values)

def load_json(data_path: str, annot_path: str) -> List:
    f = open(data_path)
    f2 = open(annot_path)
    # chart, annot
    return json.load(f), json.load(f2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/defaults/4488.json")
    parser.add_argument("--annot_path", type=str, default="./data/annotations/4488.json")
    parser.add_argument("--optim_path", type=str, default="./data/optims")
    args = vars(parser.parse_args())

    device = 'cuda'
    image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    vit = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased")
    model = SalFormer(vit, bert).to(device)
    checkpoint = torch.load('./model/model_lr6e-5_wd1e-4.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if '.json' in args['data_path']:
        chart_json, annot_json = load_json(args['data_path'], args['annot_path'])
        bayesian_optim(chart_json, annot_json, query=annot_json['tasks'][0]['question'], optim_path=args['optim_path'], chart_name=args['data_path'].split('/')[-1].strip('.json'))
    else: # batch processing
        for data_json in os.listdir(args['data_path']):
            chart_json, annot_json = load_json(os.path.join(args['data_path'], data_json), os.path.join(args['annot_path'], data_json))
            bayesian_optim(chart_json, annot_json, query=annot_json['tasks'][0]['question'], optim_path=args['optim_path'], chart_name=data_json.strip('.json'))
