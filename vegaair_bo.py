import os, sys
import argparse
import altair as alt
import json
import torch
from PIL import Image
import numpy as np
import io
from model.model import SalFormer
from transformers import AutoImageProcessor, AutoTokenizer, BertModel, SwinModel
#import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import cv2
from svg.path import parse_path
from xml.dom import minidom
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.sampling import draw_sobol_samples
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf

def predict(ques: str) -> np.ndarray:
    # Read the image bytes as an image
    image = Image.open('chart.png').convert("RGB")
    img_pt = image_processor(image, return_tensors="pt").to(device)
    inputs = tokenizer(ques, return_tensors="pt").to(device)

    mask = model(img_pt['pixel_values'], inputs)
    mask = mask.detach().cpu().squeeze().numpy()
    heatmap = (mask * 255).astype(np.uint8)

    image_np = np.array(image)
    # image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2RGBA)
    heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
    return heatmap

def update_chart(params: list, filename: str = 'chart') -> np.ndarray:
    # params: [aspect_ratio, font_size_y_label, font_size_mark, bar_size, highlight_bar_color_r, highlight_bar_color_g, highlight_bar_color_b]
    chart_json['vconcat'][0]['height'] = chart_json['vconcat'][0]['width'] * params[0]
    chart_json['vconcat'][0]['encoding']['y']['axis']['labelFontSize'] = params[1]
    chart_json['vconcat'][0]['layer'][1]['mark']['fontSize'] = params[2]
    chart_json['vconcat'][0]['layer'][0]['encoding']['size']['value'] = params[3]
    chart_json['vconcat'][0]['layer'][0]['encoding']['color']['condition']['test'] = "datum.Entity === 'UK'"
    chart_json['vconcat'][0]['layer'][0]['encoding']['color']['condition']['value'] = mcolors.to_hex([params[4], params[5], params[6]])

    chart = alt.Chart.from_json(json.dumps(chart_json))
    chart.save(f'{filename}.png')
    chart.save(f'{filename}.svg')
    return get_bbox(f'{filename}.svg')

def get_bbox(svg_file):
    xmldoc = minidom.parse(svg_file)
    PNT = xmldoc.getElementsByTagName("path")
    GROUP = xmldoc.getElementsByTagName("g")
    for g in GROUP:
        if g.getAttribute('class') == "mark-text role-axis-title":
            child = g.firstChild
            if 'rotate(-90)' in child.getAttribute('transform'): # This is the Y-axis label
                x_offset = float(child.getAttribute('transform')[11:19]) + float(child.getAttribute('font-size')[0:1]) # offset caused by axis labels
                # print(x_offset)

    for element in PNT:
        if element.getAttribute('aria-label') == "Importance: 21; Entity: UK":
            path_string = element.getAttribute('d')
            path = parse_path(path_string)
            bbox = path.boundingbox()
            bbox[0] += (x_offset - 20)
            bbox[1] += (-20)
            bbox[2] += (x_offset + 20)
            bbox[3] += (20)

    xmldoc.unlink()
    return np.asarray(bbox, dtype=int)

def optim_func(heatmap: np.array, bbox: np.array) -> float:
    return np.mean(heatmap[bbox[1]:bbox[3], bbox[0]:bbox[2]])


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
    print(chart_annotation_json['tasks'][0]['question'])

    device = 'cuda'
    image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    vit = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased")
    model = SalFormer(vit, bert).to(device)
    checkpoint = torch.load('./model/model_lr6e-5_wd1e-4.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    tkwargs = {"device": "cpu:0", "dtype": torch.double}
    bounds = torch.tensor([[[0.2], [5], [5], [10], [0], [0], [0]],\
                           [[2],  [25],[25], [40], [1], [1], [1]]], **tkwargs) # lower bound, upper bound
    x_obs = draw_sobol_samples(bounds=bounds, n=5, q=1, seed=0).squeeze(-1)
    y_obs = torch.empty(0,1)

    #Initial observations
    for x in x_obs:
        bbox = update_chart(x.tolist())
        heatmap = predict(chart_annotation_json['tasks'][0]['question'])
        y_obs = torch.concat([y_obs, torch.tensor([optim_func(heatmap, bbox)]).unsqueeze(-1)], dim=0)

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
            bbox = update_chart(candidate.tolist()[0], 'chart_best')
            # print(candidate)  # tensor([[0.2981, 0.2401]], dtype=torch.float64)

        bbox = update_chart(candidate.tolist()[0])
        heatmap = predict(chart_annotation_json['tasks'][0]['question'])
        y_obs = torch.concat([y_obs, torch.tensor([optim_func(heatmap, bbox)]).unsqueeze(-1)], dim=0)
        x_obs = torch.concat([x_obs, candidate], dim=0)
