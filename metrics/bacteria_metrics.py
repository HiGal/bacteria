import base64
import cv2
import json
import numpy as np
import os
import pandas as pd

DATASET_PATH = '../data/test'

FILE_FILTER = '.json'

LABELS = ['staphylococcus_epidermidis', 'klebsiella_pneumoniae', 'staphylococcus_aureus', 'moraxella_catarrhalis',
          'c_kefir', 'ent_cloacae']

label_metrics = np.zeros((len(LABELS), len(LABELS)), int)
seg_metrics = []
df = pd.read_csv('../data/bacteria.csv')


def set_metrics(filename):
    global label_metrics, seg_metrics, df

    with open(os.path.join(DATASET_PATH, filename), 'r') as f:
        layout = json.load(f)
    h, w = layout['imageHeight'], layout['imageWidth']
    true_mask = np.zeros((h, w), np.uint8)
    label = layout['shapes'][0]['label']

    for shape in layout['shapes']:
        polygon = np.array([point[::-1] for point in shape['points']])
        cv2.fillPoly(true_mask, [polygon], 255)

    ind = int(filename[:-len(FILE_FILTER)]) - 1
    new_label = df.at[ind, 'class']
    label_metrics[LABELS.index(new_label)][LABELS.index(label)] += 1

    with open('tmp_bacteria.png', 'wb') as fp:
        fp.write(base64.b64decode(df.at[ind, 'base64 encoded PNG (mask)'].encode()))
    mask = cv2.imread('tmp_bacteria.png', 0)
    seg_metrics += [np.count_nonzero(np.logical_and(true_mask, mask)) /
                    np.count_nonzero(np.logical_or(true_mask, mask))]


def calculate_metrics():

    mean_iou = np.mean(seg_metrics)

    precisions = dict.fromkeys(LABELS, 0.)
    for label in LABELS:
        i = LABELS.index(label)
        precisions[label] = label_metrics[i][i] / np.sum(label_metrics[i, :])
    mean_precision = np.mean(list(precisions.values()))

    score = mean_iou + np.sum(list(precisions.values()))

    print(f'mean_iou: {mean_iou}')
    for k, v in precisions.items():
        print(f'precision_{k}: {v}')
    print(f'mean_precision: {mean_precision}\nscore: {score}')


def main():
    files = os.listdir(DATASET_PATH)
    print(files)
    for file in filter(lambda x: x[-len(FILE_FILTER):].lower() == FILE_FILTER, files):
        set_metrics(file)
    calculate_metrics()


if __name__ == '__main__':
    main()
