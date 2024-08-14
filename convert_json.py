import os, sys
import argparse
import json
import re
from PIL import Image
import numpy as np
from tqdm import trange
from utils import save_chart_batch

def load_base_json(base_json):
    f = open(base_json)
    base_json = json.load(f)
    return base_json.copy()

def process_json(input_path, subset, output_path, base_path):
    for i in trange(len(os.listdir(os.path.join(input_path, subset, 'annotations')))):
        filename = os.listdir(os.path.join(input_path, subset, 'annotations'))[i]
        if not filename.endswith(".json") or filename.startswith("two_col") or filename.startswith("multi_col"): continue
        f1 = open(os.path.join(input_path, subset, f'{subset}_human.json'))
        ques_json = json.load(f1)
        questions = [x for x in ques_json if x["imgname"]==filename.replace('.json', '.png')]
        if not questions: continue
        f2 = open(os.path.join(input_path, subset, 'annotations', filename))
        annot_json = json.load(f2)
        annot_json['tasks'] = []
        for q in questions:
            annot_json['tasks'].append({"question": q['query'], "label": q['label'], "entity": '', "aria-label": ''})
        if annot_json['type'] == 'h_bar':
            output_json = load_base_json(os.path.join(base_path, 'hbar_template.json'))
            # if 'title' in annot_json['general_figure_info'].keys():
            #     output_json['vconcat'][0]['title'] = annot_json['general_figure_info']['title']['text']
            # else:
            #     output_json['vconcat'][0]['title'] = ''
            if isinstance(annot_json['models'], list) and annot_json['models'][0]:
                data_entries = []
                for i, x_label in enumerate(annot_json['models'][0]['x']):
                    value = re.sub('[^0-9]','', annot_json['models'][0]['y'][i])
                    data_entries.append({"Entity": x_label, "value": value})
                    for j in range(2):
                        if x_label in annot_json['tasks'][j]['label'] or annot_json['tasks'][j]['label'] in x_label or x_label in annot_json['tasks'][j]['question']:
                            annot_json['tasks'][j]['entity'] = x_label
                            annot_json['tasks'][j]['aria-label'] = f"value: {value}; Entity: {x_label}"
                if annot_json['tasks'][0]['entity'] or annot_json['tasks'][1]['entity']:
                    output_json['vconcat'][0]['data']['values'] = data_entries
                    output_json['name'] = filename
                    save_chart_batch(output_json, annot_json, output_path, filename.strip('.json'))
        elif annot_json['type'] == 'v_bar':
            output_json = load_base_json(os.path.join(base_path, 'vbar_template.json'))
            if isinstance(annot_json['models'], list) and annot_json['models'][0]:
                data_entries = []
                for i, x_label in enumerate(annot_json['models'][0]['x']):
                    value = re.sub('[^0-9]','', annot_json['models'][0]['y'][i])
                    data_entries.append({"Entity": x_label, "value": value})
                    for j in range(2):
                        if x_label in annot_json['tasks'][j]['label'] or annot_json['tasks'][j]['label'] in x_label or x_label in annot_json['tasks'][j]['question']:
                            annot_json['tasks'][j]['entity'] = x_label
                            annot_json['tasks'][j]['aria-label'] = f"Entity: {x_label}; value: {value}"
                if annot_json['tasks'][0]['entity'] or annot_json['tasks'][1]['entity']:
                    output_json['vconcat'][0]['data']['values'] = data_entries
                    output_json['name'] = filename
                    save_chart_batch(output_json, annot_json, output_path, filename.strip('.json'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--chartqa_path", type=str, default="/netpool/homes/wangyo/Dataset/ChartQA/")
    parser.add_argument("--subset", type=str, default="train")
    parser.add_argument("--output_path", type=str, default="/netpool/homes/wangyo/Dataset/ChartQA_VegaAltAir")
    parser.add_argument("--base_path", type=str, default="./data")
    args = vars(parser.parse_args())

    process_json(args['chartqa_path'], args['subset'], args['output_path'], args['base_path'])
