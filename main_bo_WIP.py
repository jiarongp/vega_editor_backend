from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np
import io
import base64
from model.model import SalFormer
from transformers import AutoImageProcessor, AutoTokenizer, BertModel, SwinModel
import matplotlib.pyplot as plt
import cv2
import torch
from fastapi.responses import Response, FileResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost","http://localhost:8080"],  # Allow requests from all origins, you may want to restrict this
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

class QueryData(BaseModel):
    imageDataUrl: str
    question: str

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


@app.post("/upload")
async def upload_image(query: QueryData):
    # Extract base64 encoded image data
    base64_data = query.imageDataUrl.split(",")[1]

    # Decode base64 data
    image_bytes = base64.b64decode(base64_data)

    # Read the image bytes as an image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_pt = image_processor(image, return_tensors="pt").to(device)
    inputs = tokenizer(query.question, return_tensors="pt").to(device)

    mask = model(img_pt['pixel_values'], inputs)
    mask = mask.detach().cpu().squeeze().numpy()
    heatmap = (colormap(mask) * 255).astype(np.uint8)

    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2RGBA)
    heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))

    overlay = cv2.addWeighted(image_np, 1-alpha, heatmap, alpha, 0)

    image = Image.fromarray(overlay)
    with io.BytesIO() as buf:
        image.save(buf, format='PNG')
        im_bytes = buf.getvalue()

    tkwargs = {"device": "cpu:0", "dtype": torch.double}
    bounds = torch.tensor([[0.1], [2.0]], **tkwargs)
    x_obs = draw_sobol_samples(bounds=bounds, n=5, q=1, seed=0).squeeze(-2)
    y_obs = np.mean(heatmap)


    headers = {'Content-Disposition': 'inline; filename="test.png"'}
    return Response({"aspect_ratio": 1, im_bytes: im_bytes}, headers=headers, media_type='image/png')


@app.get("/")
async def main():
    return {"aspect_ratio": 1}
