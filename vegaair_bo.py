import altair as alt
import json
import torch
from PIL import Image
import numpy as np
import io
import base64
from model.model import SalFormer
from transformers import AutoImageProcessor, AutoTokenizer, BertModel, SwinModel
import matplotlib.pyplot as plt
import cv2
from svg.path import parse_path
from svg.path.path import Line
from xml.dom import minidom
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.sampling import draw_sobol_samples
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf


def predict() -> np.ndarray:
    # Read the image bytes as an image
    image = Image.open('chart.png').convert("RGB")
    img_pt = image_processor(image, return_tensors="pt").to(device)
    inputs = tokenizer(chart_annotation_json['tasks'][0]['question'], return_tensors="pt").to(device)

    mask = model(img_pt['pixel_values'], inputs)
    mask = mask.detach().cpu().squeeze().numpy()
    heatmap = (colormap(mask) * 255).astype(np.uint8)

    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2RGBA)
    heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
    return heatmap

def update_chart(aspect_ratio: float):
    chart_json['vconcat'][0]['height'] = chart_json['vconcat'][0]['width'] * aspect_ratio
    chart = alt.Chart.from_json(json.dumps(chart_json))
    chart.save('chart.png')
    chart.save('chart.svg')

def get_bbox(svg_file):
    xmldoc = minidom.parse(svg_file)
    PNT = xmldoc.getElementsByTagName("path")
    GROUP = xmldoc.getElementsByTagName("g")
    for g in GROUP:
        if g.getAttribute('class') == "mark-text role-axis-title":
            child = g.firstChild
            if 'rotate(-90)' in child.getAttribute('transform'): # This is the Y-axis label
                x_offset = float(child.getAttribute('transform')[11:19]) + float(child.getAttribute('font-size')[0:1])
                # print(x_offset)

    for element in PNT:
        if element.getAttribute('aria-label') == "Importance: 21; Entity: UK":
            path_string = element.getAttribute('d')
            path = parse_path(path_string)
            bbox = path.boundingbox()
            bbox[0] += x_offset
            bbox[2] += x_offset
            # print(bbox)

    xmldoc.unlink()
    return np.asarray(bbox, dtype=int)

def optim_func(heatmap: np.array):
    return np.mean(heatmap)


if __name__ == '__main__':
    f = open('../vega_editor/chartqa_bar20_noq/defaults/4488.json')
    chart_json = json.load(f)
    f2 = open('../vega_editor/chartqa_bar20_noq/annotations/4488.json')
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
    colormap = plt.cm.jet
    alpha = 0.5

    # image = Image.fromarray(overlay)
    # with io.BytesIO() as buf:
    #     image.save(buf, format='PNG')
    #     im_bytes = buf.getvalue()
    # #overlay = cv2.addWeighted(image_np, 1-alpha, heatmap, alpha, 0)

    tkwargs = {"device": "cpu:0", "dtype": torch.double}
    bounds = torch.tensor([[0.5], [2.0]], **tkwargs)
    x_obs = draw_sobol_samples(bounds=bounds, n=5, q=1, seed=0).squeeze(-1)
    y_obs = torch.empty(0,1)


    for x in x_obs:
        update_chart(x.item())
        heatmap = predict()
        bbox = get_bbox("chart.svg")
        print(bbox)
        print(np.shape(heatmap))
        y_obs = torch.concat([y_obs, torch.tensor([np.mean(heatmap[bbox[1]:bbox[3], bbox[0]:bbox[2], :])]).unsqueeze(-1)], dim=0)



    for i in range(20):
        print(x_obs)
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
            logNEI, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
        )
        print(candidate)  # tensor([[0.2981, 0.2401]], dtype=torch.float64)

        update_chart(candidate.tolist()[0][0])
        heatmap = predict()
        bbox = get_bbox("chart.svg")
        y_obs = torch.concat([y_obs, torch.tensor([np.mean(heatmap[bbox[1]:bbox[3], bbox[0]:bbox[2], :])]).unsqueeze(-1)], dim=0)
        x_obs = torch.concat([x_obs, candidate], dim=0)
