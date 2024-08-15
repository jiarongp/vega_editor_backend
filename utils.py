import json
import os
import numpy as np
import altair as alt
from matplotlib import colors as mcolors
from svg.path import parse_path
from xml.dom import minidom
import cv2
from typing import List
from PIL import Image

def update_chart(chart_json: json, params: list, annotation: json, filename: str = 'chart') -> np.ndarray:
    # params: [aspect_ratio, font_size_y_label, font_size_mark, bar_size, highlight_bar_color_r, highlight_bar_color_g, highlight_bar_color_b]
    chart_json['vconcat'][0]['height'] = chart_json['vconcat'][0]['width'] * params[0]
    chart_json['vconcat'][0]['layer'][1]['mark']['fontSize'] = params[2]
    chart_json['vconcat'][0]['layer'][0]['encoding']['size']['value'] = params[3]
    for i, entity in enumerate(annotation['tasks'][1]['entity']):
        chart_json['vconcat'][0]['layer'][0]['encoding']['color']['condition'].append({
            "test": f"datum.Entity === '{entity}'",
            "value": mcolors.to_hex([params[4], params[5], params[6]])
        })
        # print({
        #     "test": f"datum.Entity === '{entity}'",
        #     "value": mcolors.to_hex([params[4], params[5], params[6]])
        # })
    if annotation['type'] == 'h_bar':
        chart_json['vconcat'][0]['encoding']['y']['axis']['labelFontSize'] = params[1]
    elif annotation['type'] == 'v_bar':
        chart_json['vconcat'][0]['encoding']['x']['axis']['labelFontSize'] = params[1]

    chart = alt.Chart.from_json(json.dumps(chart_json))
    chart.save(f'data/{filename}.png')
    chart.save(f'data/{filename}.svg')
    im = Image.open(f'data/{filename}.png').convert("RGB")
    im = np.array(im)
    bboxes = get_bboxes(f'data/{filename}.svg', annotation, np.shape(im))
    for bbox in bboxes:
        cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0, 255, 0),2)
        cv2.imwrite(f'data/{filename}_bbox.png', im)
    return bboxes

def save_chart_batch(chart_json: json, annotation: json, output_path: str, filename: str = 'chart'):
    chart = alt.Chart.from_json(json.dumps(chart_json))
    chart.save(os.path.join(output_path, 'svgs', f'{filename}.svg'))
    with open(os.path.join(output_path, 'vegas', f'{filename}.json'), 'w') as out_file:
        json.dump(chart_json, out_file)
    with open(os.path.join(output_path, 'annotations', f'{filename}.json'), 'w') as out_file:
        json.dump(annotation, out_file)
    #return get_bbox(f'data/{filename}.svg', annotation)

def get_bboxes(svg_file, annotation:json, imshape: np.ndarray) -> List[np.ndarray]:
    xmldoc = minidom.parse(svg_file)
    PNT = xmldoc.getElementsByTagName("path")
    GROUP = xmldoc.getElementsByTagName("g")
    x_offset = 0
    for g in GROUP:
        if g.getAttribute('class') == "mark-text role-axis-title":
            child = g.firstChild
            # find Y-axis label in h_bar or X-axis label in v_bar
            if (annotation['type'] == 'h_bar' and 'rotate(-90)' in child.getAttribute('transform')) \
                or (annotation['type'] == 'v_bar'and not 'rotate(-90)' in child.getAttribute('transform')):
                x_offset = float(child.getAttribute('transform').strip('translate(-').split(',')[0]) + float(child.getAttribute('font-size')[0:1]) # offset caused by axis labels
                # print(x_offset)
                break

    bboxes = []
    for ariaLabel in annotation['tasks'][1]['aria-label']:
        for element in PNT:
            if element.getAttribute('aria-label') == ariaLabel:
                path_string = element.getAttribute('d')
                path = parse_path(path_string)
                bbox = path.boundingbox()
                bbox[0] += (x_offset - 30) # making bounding box bigger
                bbox[1] += (0)
                bbox[2] += (x_offset + 50)
                bbox[3] += (40)
                if bbox[0] < 0: bbox[0] = 0
                if bbox[1] < 0: bbox[1] = 0
                if bbox[2] > imshape[1]: bbox[2] = imshape[1] - 1
                if bbox[3] > imshape[0]: bbox[3] = imshape[0] - 1
                bboxes.append(np.asarray(bbox, dtype=int))
                break

    xmldoc.unlink()
    return bboxes
