import os, sys
import argparse
import json
import re
from typing import List
from tqdm import trange
from utils.utils import save_chart_batch
from decimal import Decimal

def load_base_json(base_json: json):
    f = open(base_json)
    base_json = json.load(f)
    return base_json.copy()

def sortby_value(data_entries: List, v_type: str) -> List:
    if v_type == 'h_bar':
        sorted_data = sorted(data_entries, key=lambda x:float(x['value']), reverse=True)
        return list(sorted_data)
    elif v_type == 'v_bar':
        return data_entries

def write_tasks(annot_json: json, questions: List, input_path: str, base_path: str, output_path: str, filename:str, v_type: str):
    if v_type == 'h_bar':
        output_json = load_base_json(os.path.join(base_path, 'hbar_template.json'))
    elif v_type == 'v_bar':
        output_json = load_base_json(os.path.join(base_path, 'vbar_template.json'))
    else:
        # print('type not supported')
        return

    if not isinstance(annot_json['models'], list) or not annot_json['models'][0] or not len(annot_json['models']) == 1: return # Now only support single bar chart
    annot_json['tasks'], data_entries = [], []
    # if 'title' in annot_json['general_figure_info'].keys():
    #     output_json['vconcat'][0]['title'] = annot_json['general_figure_info']['title']['text']
    # else:
    #     output_json['vconcat'][0]['title'] = ''
    for ii, q in enumerate(questions):
        entities, ariaLabels = [], []
        q['label'] = re.sub('[{()}]', '', q['label'])
        q_labels = q['label'].split(',')
        for q_label in q_labels:
            lowest_label = ''
            lowest_value = 100000000
            highest_label = ''
            highest_value = -1
            for i, x_l in enumerate(annot_json['models'][0]['x']):
                x_label = x_l.replace("'", "")
                value = re.sub('[^0-9.]','', annot_json['models'][0]['y'][i])
                try:
                    float(value)
                except ValueError:
                    return
                if not value: return
                value = str(Decimal(value)).replace('.0','')
                if value.find('0.') > -1:
                    if len(value) - value.find('.') > 2:
                        value = str(Decimal(value).quantize(Decimal('.01'))) # round up to 2 decimal places
                elif value.find('.') > -1:
                    if len(value) - value.find('.') > 1:
                        value = str(Decimal(value).quantize(Decimal('.1'))) # round up to 1 decimal places
                if ii == 0:
                    data_entries.append({"Entity": x_label, "value": str(Decimal(value)).replace('.0','')})

                if x_label.lower() in q_label.lower() or q_label.lower() in x_label.lower() or x_label.lower() in q['query'].lower():
                    entities.append(x_label)
                    if v_type == 'h_bar':
                        ariaLabels.append(f"value: {str(Decimal(value)).replace('.0','')}; Entity: {x_label}")
                    elif v_type == 'v_bar':
                        ariaLabels.append(f"Entity: {x_label}; value: {str(Decimal(value)).replace('.0','')}")

                if float(value) < float(lowest_value):
                    lowest_label = x_label
                    lowest_value = value
                if float(value) > float(highest_value):
                    highest_label = x_label
                    highest_value = value
            if lowest_label and ('least' in q['query'].lower() or 'lowest' in q['query'].lower()):
                entities.append(lowest_label)
                if v_type == 'h_bar':
                    ariaLabels.append(f"value: {lowest_value}; Entity: {lowest_label}")
                elif v_type == 'v_bar':
                    ariaLabels.append(f"Entity: {lowest_label}; value: {lowest_value}")
            if highest_label and ('most' in q['query'].lower() or 'highest' in q['query'].lower()):
                entities.append(highest_label)
                if v_type == 'h_bar':
                    ariaLabels.append(f"value: {highest_value}; Entity: {highest_label}")
                elif v_type == 'v_bar':
                    ariaLabels.append(f"Entity: {highest_label}; value: {highest_value}")

            if len(entities) > 0:
                annot_json['tasks'].append({"question": q['query'], "labels": q_labels, "entity": entities, "aria-label": ariaLabels})

        if len(annot_json['tasks']) > 0:    
            data_entries = sortby_value(data_entries, v_type) # sort h_bars
            output_json['vconcat'][0]['data']['values'] = data_entries
            output_json['name'] = filename
            save_chart_batch(output_json, annot_json, input_path, output_path, filename.strip('.json'))

def process_json(input_path: str, subset: str, output_path: str, base_path: str):
    for i in trange(len(os.listdir(os.path.join(input_path, subset, 'annotations')))):
        filename = os.listdir(os.path.join(input_path, subset, 'annotations'))[i]
        if not filename.endswith(".json") or filename.startswith("two_col") or filename.startswith("multi_col"): continue
        f1 = open(os.path.join(input_path, subset, f'{subset}_human.json'))
        ques_json = json.load(f1)
        questions = [x for x in ques_json if x["imgname"]==filename.replace('.json', '.png')]
        if not questions: continue
        f2 = open(os.path.join(input_path, subset, 'annotations', filename))
        annot_json = json.load(f2)
        write_tasks(annot_json, questions, os.path.join(input_path,subset), base_path, output_path, filename, annot_json['type'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--chartqa_path", type=str, default="/netpool/homes/wangyo/Dataset/ChartQA/")
    parser.add_argument("--subset", type=str, default="train")
    parser.add_argument("--output_path", type=str, default="/netpool/homes/wangyo/Dataset/ChartQA_VegaAltAir")
    parser.add_argument("--base_path", type=str, default="./data")
    args = vars(parser.parse_args())

    process_json(args['chartqa_path'], args['subset'], args['output_path'], args['base_path'])
