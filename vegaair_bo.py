import os, sys
import argparse
import json
import torch
from PIL import Image
import numpy as np
from typing import List
from model.model import SalFormer
from transformers import AutoImageProcessor, AutoTokenizer, BertModel, SwinModel
import cv2
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.sampling import draw_sobol_samples
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from utils import update_chart
from metrics import wave_metric

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

    image_np = np.array(image)
    # image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2RGBA)
    heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
    return [heatmap, wave_metric(image)]

def optim_func(predictions: List, bbox: np.array) -> float:
    """
    Optimisation function of BO.

    Args:
        predictions[list]: same as the output of predict()
        bbox: bounding box coordinates

    Returns: score of the optimisation function
        
    """
    WAVE = predictions[1] * 255.
    bbox_heatmap = np.mean(predictions[0][bbox[1]:bbox[3], bbox[0]:bbox[2]])

    return np.mean(0.2 * WAVE + 0.8 * bbox_heatmap)

def bayesian_optim(chart_json: json, annotation:json):
    tkwargs = {"device": "cpu:0", "dtype": torch.double}
    bounds = torch.tensor([[[0.2], [5], [5], [10], [0], [0], [0]],\
                           [[2],  [25],[25], [40], [1], [1], [1]]], **tkwargs) # lower bound, upper bound
    x_obs = draw_sobol_samples(bounds=bounds, n=5, q=1, seed=0).squeeze(-1)
    y_obs = torch.empty(0,1)

    #Initial observations
    for x in x_obs:
        bbox = update_chart(chart_json, x.tolist(), annotation)
        predictions = predict(annotation['tasks'][0]['question'])
        y_obs = torch.concat([y_obs, torch.tensor([optim_func(predictions, bbox)]).unsqueeze(-1)], dim=0)

    y_max = 0
    #Optimization loop
    for i in range(50):
        print(y_obs)
        gp = SingleTaskGP(
            train_X=x_obs,
            train_Y=y_obs,
            # input_transform=Normalize(d=1),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        logNEI = LogExpectedImprovement(model=gp, best_f=y_obs.max())
        candidate, acq_value = optimize_acqf(
            logNEI, bounds=bounds.squeeze(-1), q=1, num_restarts=5, raw_samples=20,
        )
        if y_max < y_obs[-1].item():
            y_max = y_obs[-1].item()
            bbox = update_chart(chart_json, candidate.tolist()[0], annotation, 'chart_best')
            # print(candidate)  # tensor([[0.2981, 0.2401]], dtype=torch.float64)

        bbox = update_chart(chart_json, candidate.tolist()[0], annotation)
        predictions = predict(annotation['tasks'][0]['question'])
        y_obs = torch.concat([y_obs, torch.tensor([optim_func(predictions, bbox)]).unsqueeze(-1)], dim=0)
        x_obs = torch.concat([x_obs, candidate], dim=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_json", type=str, default="../vega_editor/chartqa_bar20_noq/defaults/4488.json")
    parser.add_argument("--annot_json", type=str, default="../vega_editor/chartqa_bar20_noq/annotations/4488.json")
    parser.add_argument('--process_csv', action='store_true')
    args = vars(parser.parse_args())

    f = open(args['data_json'])
    chart_json = json.load(f)
    f2 = open(args['annot_json'])
    chart_annotation_json = json.load(f2)

    device = 'cuda'
    image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    vit = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased")
    model = SalFormer(vit, bert).to(device)
    checkpoint = torch.load('./model/model_lr6e-5_wd1e-4.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    bayesian_optim(chart_json, chart_annotation_json)
