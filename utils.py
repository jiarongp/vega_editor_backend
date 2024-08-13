import json
import numpy as np
import altair as alt
from matplotlib import colors as mcolors
from svg.path import parse_path
from xml.dom import minidom

def update_chart(chart_json: json, params: list, annotation: json, filename: str = 'chart') -> np.ndarray:
    # params: [aspect_ratio, font_size_y_label, font_size_mark, bar_size, highlight_bar_color_r, highlight_bar_color_g, highlight_bar_color_b]
    chart_json['vconcat'][0]['height'] = chart_json['vconcat'][0]['width'] * params[0]
    chart_json['vconcat'][0]['encoding']['y']['axis']['labelFontSize'] = params[1]
    chart_json['vconcat'][0]['layer'][1]['mark']['fontSize'] = params[2]
    chart_json['vconcat'][0]['layer'][0]['encoding']['size']['value'] = params[3]
    chart_json['vconcat'][0]['layer'][0]['encoding']['color']['condition']['test'] = f"datum.Entity === '{annotation['entity']}'"
    chart_json['vconcat'][0]['layer'][0]['encoding']['color']['condition']['value'] = mcolors.to_hex([params[4], params[5], params[6]])

    chart = alt.Chart.from_json(json.dumps(chart_json))
    chart.save(f'{filename}.png')
    chart.save(f'{filename}.svg')
    return get_bbox(f'{filename}.svg', annotation)

def get_bbox(svg_file, annotation:json) -> np.ndarray:
    xmldoc = minidom.parse(svg_file)
    PNT = xmldoc.getElementsByTagName("path")
    GROUP = xmldoc.getElementsByTagName("g")
    for g in GROUP:
        if g.getAttribute('class') == "mark-text role-axis-title":
            child = g.firstChild
            if 'rotate(-90)' in child.getAttribute('transform'): # This is the Y-axis label
                # print(child.getAttribute('transform'))
                x_offset = float(child.getAttribute('transform')[11:19]) + float(child.getAttribute('font-size')[0:1]) # offset caused by axis labels
                # print(x_offset)

    for element in PNT:
        if element.getAttribute('aria-label') == annotation['aria-label']:
            path_string = element.getAttribute('d')
            path = parse_path(path_string)
            bbox = path.boundingbox()
            bbox[0] += (x_offset - 20) # making bounding box bigger
            bbox[1] += (-20)
            bbox[2] += (x_offset + 20)
            bbox[3] += (20)

    xmldoc.unlink()
    return np.asarray(bbox, dtype=int)
