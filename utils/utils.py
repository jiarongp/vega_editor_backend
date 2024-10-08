import json
import os
import shutil
import numpy as np
import altair as alt
from matplotlib import colors as mcolors
from svg.path import parse_path
from xml.dom import minidom
import cv2
from typing import List
from PIL import Image
from utils.visual_density import vd_loss

PARAM_BOUNDS = {
    'aspect_ratio': [0.5, 2],
    'font_size_axis': [10, 32],
    'font_size_mark': [10, 32],
    'bar_size': [30, 150],
    'axis_label_rotation': [-90, -45, 0],
}

def calc_param(dict_key: str, param: float, is_discrete: bool = False) -> float:
    if is_discrete:
        return PARAM_BOUNDS[dict_key][int(param)]
    return PARAM_BOUNDS[dict_key][0] + (PARAM_BOUNDS[dict_key][1] - PARAM_BOUNDS[dict_key][0]) * param

def change_orientation(chart_json: json, annotation: json, ot: int) -> json:
    # 0: horizontal, 1: vertical
    if (ot == 0 and annotation['type'] == 'h_bar') or (ot == 1 and annotation['type'] == 'v_bar'):
        print('not changing orientation 1')
        return chart_json, annotation
    elif ot == 0 and annotation['type'] == 'v_bar':
        annotation['type'] = 'h_bar'
        chart_json['vconcat'][0]['encoding']['tmp'] = chart_json['vconcat'][0]['encoding']['x']
        chart_json['vconcat'][0]['encoding']['x'] = chart_json['vconcat'][0]['encoding']['y']
        chart_json['vconcat'][0]['encoding']['y'] = chart_json['vconcat'][0]['encoding']['tmp']
        del chart_json['vconcat'][0]['encoding']['tmp']
        chart_json['vconcat'][0]['layer'][1]['encoding']['x'] = chart_json['vconcat'][0]['layer'][1]['encoding']['y']
        del chart_json['vconcat'][0]['layer'][1]['encoding']['y']
        for i, task in enumerate(annotation['tasks']):
            for j, label in enumerate(task['aria-label']):
                parts = label.split(';')
                annotation['tasks'][i]['aria-label'][j] = f"{parts[1].strip()}; {parts[0].strip()}"
        return chart_json, annotation
    elif ot == 1 and annotation['type'] == 'h_bar':
        annotation['type'] = 'v_bar'
        chart_json['vconcat'][0]['encoding']['tmp'] = chart_json['vconcat'][0]['encoding']['x']
        chart_json['vconcat'][0]['encoding']['x'] = chart_json['vconcat'][0]['encoding']['y']
        chart_json['vconcat'][0]['encoding']['y'] = chart_json['vconcat'][0]['encoding']['tmp']
        del chart_json['vconcat'][0]['encoding']['tmp']
        chart_json['vconcat'][0]['layer'][1]['encoding']['y'] = chart_json['vconcat'][0]['layer'][1]['encoding']['x']
        del chart_json['vconcat'][0]['layer'][1]['encoding']['x']
        for i, task in enumerate(annotation['tasks']):
            for j, label in enumerate(task['aria-label']):
                parts = label.split(';')
                annotation['tasks'][i]['aria-label'][j] = f"{parts[1].strip()}; {parts[0].strip()}"
        return chart_json, annotation

def write_text(im: np.ndarray, text: str):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (20,20)
    fontScale              = 1
    fontColor              = (0,0,0)
    thickness              = 1
    lineType               = 2

    cv2.putText(im, text, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)

def update_chart(chart_json: json, params: dict, annotation: json, data_path: str = 'data', filename: str = 'chart') -> np.ndarray:
    # PARAMS: [aspect_ratio, font_size_axis, font_size_mark, bar_size, highlight_bar_color_r, highlight_bar_color_g, highlight_bar_color_b, axis_label_rotation(v_bar), orientation]
    chart_json, annotation = change_orientation(chart_json, annotation, params['x_ot']) # change_orientation of the chart (horinzontal or vertical)

    chart_json['vconcat'][0]['height'] = chart_json['vconcat'][0]['width'] * calc_param('aspect_ratio', params['x0'])
    chart_json['vconcat'][0]['layer'][1]['mark']['fontSize'] = calc_param('font_size_mark', params['x2'])
    chart_json['vconcat'][0]['layer'][0]['encoding']['size']['value'] = calc_param('bar_size', params['x3'])
    color_rgb = [params['x4'], params['x5'], params['x6']]
    for _, entity in enumerate(annotation['tasks'][0]['entity']):
        f = False
        for dd in chart_json['vconcat'][0]['layer'][0]['encoding']['color']['condition']:
            if dd['test'] == f"datum.Entity === '{entity}'":
                dd['value'] = mcolors.to_hex(color_rgb)
                f = True
                break
        if f: continue
        chart_json['vconcat'][0]['layer'][0]['encoding']['color']['condition'].append({
            "test": f"datum.Entity === '{entity}'",
            "value": mcolors.to_hex(color_rgb)
        })
    if annotation['type'] == 'h_bar':
        chart_json['vconcat'][0]['encoding']['y']['axis']['labelFontSize'] = calc_param('font_size_axis', params['x1'])
        chart_json['vconcat'][0]['layer'][1]['mark']['xOffset'] = calc_param('font_size_axis', params['x1'])/2
        chart_json['vconcat'][0]['layer'][1]['mark']['yOffset'] = 0
        chart_json['vconcat'][0]['layer'][1]['mark']['dx'] = 16
        chart_json['vconcat'][0]['layer'][1]['mark']['align'] = 'left'
        chart_json['vconcat'][0]['encoding']['y']['axis']['labelAngle'] = 0
    elif annotation['type'] == 'v_bar':
        chart_json['vconcat'][0]['encoding']['x']['axis']['labelFontSize'] = calc_param('font_size_axis', params['x1'])
        chart_json['vconcat'][0]['layer'][1]['mark']['xOffset'] = 0
        chart_json['vconcat'][0]['layer'][1]['mark']['yOffset'] = -calc_param('font_size_axis', params['x1'])*2/3
        chart_json['vconcat'][0]['layer'][1]['mark']['dx'] = 0
        chart_json['vconcat'][0]['layer'][1]['mark']['align'] = 'center'
        chart_json['vconcat'][0]['encoding']['x']['axis']['labelAngle'] = calc_param('axis_label_rotation', params['x_rt'], is_discrete=True)

    chart = alt.Chart.from_json(json.dumps(chart_json))
    chart.save(f'{data_path}/{filename}.png')
    chart.save(f'{data_path}/{filename}.svg')
    im = Image.open(f'{data_path}/{filename}.png').convert("RGB")
    im_np = np.array(im)
    bboxes = get_bboxes(f'{data_path}/{filename}.svg', annotation, np.shape(im_np))

    if not filename == 'chart':
        # print(filename)
        with open(f'{data_path}/{filename}.json', 'w') as out_file:
            json.dump(chart_json, out_file)
        for bbox in bboxes:
            cv2.rectangle(im_np,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0, 255, 0), 2)
        # DEBUG: print visual density
        #im_grey = np.array(im.convert("L"))
        # write_text(im_np, str(vd_loss(im_grey)))
        cv2.imwrite(f'{data_path}/{filename}_bbox.png', im_np)
    return bboxes

def save_chart_batch(chart_json: json, annotation: json, input_path: str, output_path: str, filename: str = 'chart'):    
    chart = alt.Chart.from_json(json.dumps(chart_json))
    chart.save(os.path.join(output_path, 'svgs', f'{filename}.svg'))
    with open(os.path.join(output_path, 'vegas', f'{filename}.json'), 'w') as out_file:
        json.dump(chart_json, out_file)
    with open(os.path.join(output_path, 'annotations', f'{filename}.json'), 'w') as out_file:
        json.dump(annotation, out_file)
    shutil.copy(os.path.join(input_path, 'png', filename+'.png'), os.path.join(output_path, 'png', filename+'.png'))
    shutil.copy(os.path.join(input_path, 'tables', filename+'.csv'), os.path.join(output_path, 'tables', filename+'.csv'))

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
            or (annotation['type'] == 'v_bar' and not 'rotate(-90)' in child.getAttribute('transform')):
                x_offset = float(child.getAttribute('transform').strip('translate(-').split(',')[0]) + float(child.getAttribute('font-size')[0:1]) # offset caused by axis labels
                # print(x_offset)
                break

    bboxes = []
    for ariaLabel in annotation['tasks'][0]['aria-label']:
        for element in PNT:
            if element.getAttribute('aria-label') == ariaLabel:
                path_string = element.getAttribute('d')
                path = parse_path(path_string)
                bbox = path.boundingbox()
                if annotation['type'] == 'h_bar':
                    bbox[0] += (x_offset - 120) # making bounding box bigger
                    bbox[1] += (0)
                    bbox[2] += (x_offset + 120)
                    bbox[3] += (50)
                if bbox[0] < 0: bbox[0] = 0
                if bbox[1] < 0: bbox[1] = 0
                if bbox[2] > imshape[1]: bbox[2] = imshape[1] - 1
                if bbox[3] > imshape[0]: bbox[3] = imshape[0] - 1
                bboxes.append(np.asarray(bbox, dtype=int))
                break

    xmldoc.unlink()
    return bboxes

def load_json(data_path: str, annot_path: str) -> List:
    f = open(data_path)
    f2 = open(annot_path)
    # chart, annot
    return json.load(f), json.load(f2)
