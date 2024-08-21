import os, sys
import argparse
import json
import torch
torch.manual_seed(42)
from tqdm import trange
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
from utils.utils import update_chart
from utils.visual_density import vd_loss
from utils.metrics import wave_metric

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
    heatmap = cv2.resize(heatmap, (image.size[1], image.size[0]))
    return [heatmap, image, np.array(gary_image)]

def optim_func(predictions: List, bboxes: List[np.ndarray]) -> float:
    """
    Optimisation function of BO.

    Args:
        predictions[list]: same as the output of predict()
        bbox: bounding box coordinates

    Returns: score of the optimisation function
    """
    # WAVE is a metric that measures how close the colors in the heatmap are to the preferred colors from human [0, 1]
    WAVE = wave_metric(predictions[1]) * 255.
    # heatmap_mean is the mean value of saliency maps in the bounding box (larger than 32, which thresholds the whitespaces out)
    heatmap_mean = 0
    for bbox in bboxes:
        bbox_heapmap = predictions[0][bbox[1]:bbox[3], bbox[0]:bbox[2]]
        if bbox_heapmap[bbox_heapmap>8].size > 0:
            heatmap_mean += np.mean(bbox_heapmap[bbox_heapmap>8]) # thresholding the low salient pixels, so that the size of bounding box won't matter that much
    return np.mean(WAVE + 2 * heatmap_mean / len(bboxes) - 512 * vd_loss(predictions[2])) # 0.596 is the average VD of ChartQA

def bayesian_optim(chart_json: json, annotation:json, query: str, optim_path: str, chart_name:str):
    tkwargs = {"device": "cpu:0", "dtype": torch.double}
    bounds = torch.tensor([[[0.5], [10], [10], [20], [0], [0], [0]],\
                           [[2],  [36],[36], [120], [1], [1], [1]]], **tkwargs) # lower bound, upper bound
    x_obs = draw_sobol_samples(bounds=bounds, n=5, q=1, seed=0).squeeze(-1)
    y_obs = torch.empty(0,1)

    #Initial observations
    for x in x_obs:
        bboxes = update_chart(chart_json, x.tolist(), annotation)
        if len(bboxes) == 0:
            # print('no valid bounding boxes found')
            return
        predictions = predict(query)
        y_obs = torch.concat([y_obs, torch.tensor([optim_func(predictions, bboxes)]).unsqueeze(-1)], dim=0)

    #print(y_obs, x_obs)
    y_max = 0
    # Optimization loop
    max_iter = 50
    best_iter = 0
    for i in trange(max_iter):
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
        if y_max < y_obs[-1].item(): # Found a better solution
            y_max = y_obs[-1].item()
            best_iter = i
            update_chart(chart_json, candidate.tolist()[0], annotation, optim_path, chart_name)

        bboxes = update_chart(chart_json, candidate.tolist()[0], annotation)
        predictions = predict(query)
        y_obs = torch.concat([y_obs, torch.tensor([optim_func(predictions, bboxes)]).unsqueeze(-1)], dim=0)
        x_obs = torch.concat([x_obs, candidate], dim=0)

    print('best iter appears at', best_iter)

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
