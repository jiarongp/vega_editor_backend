import os
import json
import argparse
from PIL import Image

from pytesseract import pytesseract
pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

def txt_loss(img: Image, chart_json: json) -> float:
    ocr_all = pytesseract.image_to_string(img).lower()
    cnt = 0
    for entry in chart_json['vconcat'][0]['data']['values']:
        if entry['Entity'].lower() in ocr_all:
            cnt += 1
        if entry['value'].lower() in ocr_all:
            cnt += 1
    return cnt / (len(chart_json['vconcat'][0]['data']['values']) * 2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, default="/netpool/homes/wangyo/Projects/vega_editor_backend/data/chart.png")
    args = vars(parser.parse_args())

    print(pytesseract.image_to_string(Image.open(args['img_path'])))
