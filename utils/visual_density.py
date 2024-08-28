import os
import numpy as np
import argparse
from PIL import Image

# Visual Density is a metric that measures the area of inks used in the chart [0, 1]    
def vd_loss(img_arr: np.ndarray) -> float:
    bg_ratio = img_arr[img_arr>253].size / img_arr.size
    M = 0.5956 # the average VD of ChartQA
    STD = 0.0926 #  the std VD of ChartQA
    if bg_ratio < M + 2 * STD and bg_ratio > M - 2 * STD:
        return 0
    return np.abs(bg_ratio - M)

# this is the loss to aviod overlaps of neighbor bars
def overlap_loss(bandwidth: float, im_height: float, bar_num: int) -> float:
    if bar_num * bandwidth > im_height:
        return 1
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/netpool/homes/wangyo/Dataset/ChartQA/train/png")
    args = vars(parser.parse_args())

    VDs = []
    for data_json in os.listdir(args['data_path']):
        image = Image.open(os.path.join(args['data_path'], data_json)).convert("RGB")
        gary_image = np.array(image.convert('L'))
        VDs.append(gary_image[gary_image>253].size / gary_image.size)
    print(np.mean(VDs), np.std(VDs))
    # 0.5955746649719542 0.09262844232515328
