import os, sys
import argparse
import json
from PIL import Image
import numpy as np
import io

def process_json(input_path, subset, output_path, base_path):
    for filename in os.listdir(os.path.join(input_path, subset, 'annotations')):
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
            base_json = os.path.join(base_path, 'hbar_template.json')
            f = open(base_json)
            base_json = json.load(f)
            output_json = base_json.copy()
            # if 'title' in annot_json['general_figure_info'].keys():
            #     output_json['vconcat'][0]['title'] = annot_json['general_figure_info']['title']['text']
            # else:
            #     output_json['vconcat'][0]['title'] = ''
            if isinstance(annot_json['models'], list) and annot_json['models'][0]:
                data_entries = []
                for i, x_label in enumerate(annot_json['models'][0]['x']):
                    data_entries.append({"y": x_label, "x": annot_json['models'][0]['y'][i]})
                output_json['vconcat'][0]['data']['values'] = data_entries
                output_json['name'] = filename
                with open(os.path.join(output_path, 'vegas', filename), 'w') as out_file:
                    json.dump(output_json, out_file)
                with open(os.path.join(output_path, 'annotations', filename), 'w') as out_file:
                    json.dump(annot_json, out_file)
        elif annot_json['type'] == 'v_bar':
            base_json = os.path.join(base_path, 'vbar_template.json')
            f = open(base_json)
            base_json = json.load(f)
            output_json = base_json.copy()
            if isinstance(annot_json['models'], list) and annot_json['models'][0]:
                data_entries = []
                for i, x_label in enumerate(annot_json['models'][0]['x']):
                    data_entries.append({"x": x_label, "y": annot_json['models'][0]['y'][i]})
                output_json['vconcat'][0]['data']['values'] = data_entries
                output_json['name'] = filename
                with open(os.path.join(output_path, 'vegas', filename), 'w') as out_file:
                    json.dump(output_json, out_file)
                with open(os.path.join(output_path, 'annotations', filename), 'w') as out_file:
                    json.dump(annot_json, out_file)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--chartqa_path", type=str, default="/netpool/homes/wangyo/Dataset/ChartQA/")
    parser.add_argument("--subset", type=str, default="train")
    parser.add_argument("--output_path", type=str, default="/netpool/homes/wangyo/Dataset/ChartQA_VegaAltAir")
    parser.add_argument("--base_path", type=str, default="/netpool/homes/wangyo/Projects/vega_editor/chartqa_bar20_noq/defaults")
    args = vars(parser.parse_args())

    process_json(args['chartqa_path'], args['subset'], args['output_path'], args['base_path'])
