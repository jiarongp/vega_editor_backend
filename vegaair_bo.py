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

f = open('../vega_editor/chartqa_bar20_noq/defaults/4488.json')
chart_json = json.load(f)
#print(chart_json)


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


from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.utils.sampling import draw_sobol_samples

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
    chart_json['vconcat'][0]['height'] = chart_json['vconcat'][0]['width'] * x.item()
    chart = alt.Chart.from_json(json.dumps(chart_json))
    chart.save('chart.png')

    # Read the image bytes as an image
    image = Image.open('chart.png').convert("RGB")
    img_pt = image_processor(image, return_tensors="pt").to(device)
    inputs = tokenizer('what is the minimum value of this chart', return_tensors="pt").to(device)

    mask = model(img_pt['pixel_values'], inputs)
    mask = mask.detach().cpu().squeeze().numpy()
    heatmap = (colormap(mask) * 255).astype(np.uint8)

    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2RGBA)
    heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
    y_obs = torch.concat([y_obs, torch.tensor([np.mean(heatmap)]).unsqueeze(-1)], dim=0)

from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf

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

    chart_json['vconcat'][0]['height'] = chart_json['vconcat'][0]['width'] * candidate.tolist()[0][0]
    chart = alt.Chart.from_json(json.dumps(chart_json))
    chart.save('chart.png')

    # Read the image bytes as an image
    image = Image.open('chart.png').convert("RGB")
    img_pt = image_processor(image, return_tensors="pt").to(device)
    inputs = tokenizer('what is the maximum value of this chart', return_tensors="pt").to(device)

    mask = model(img_pt['pixel_values'], inputs)
    mask = mask.detach().cpu().squeeze().numpy()
    heatmap = (colormap(mask) * 255).astype(np.uint8)

    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2RGBA)
    heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
    y_obs = torch.concat([y_obs, torch.tensor([np.mean(heatmap)]).unsqueeze(-1)], dim=0)
    x_obs = torch.concat([x_obs, candidate], dim=0)
