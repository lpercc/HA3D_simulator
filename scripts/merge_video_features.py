from extract_video_features_16 import read_tsv
import glob
import sys
import re
import csv
from tqdm import tqdm
import argparse
TSV_FIELDNAMES = ["scanId", "viewpointId", "image_w", "image_h", "vfov", "features"]
sys.path.append('./')
def mergeFeatures(file_name):
    files = sorted(glob.glob(f"{file_name}*.tsv"))
    start = 0
    with open(f'{file_name}.tsv', "w") as tsv_out_file:
        writer = csv.DictWriter(tsv_out_file, delimiter="\t", fieldnames=TSV_FIELDNAMES)
        for file in files:
            print(file,start)
            split_file_name = re.split('-|\.|_', file)
            assert int(split_file_name[6]) == start
            with open(file, "rt") as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter="\t", fieldnames=TSV_FIELDNAMES)
                bar = tqdm(reader)
                for item in reader:
                    writer.writerow(item)
                    bar.set_description(f'{file} iter feature')
            start = int(split_file_name[7])
def read_feature(file_name):
    with open(f"{file_name}.tsv", 'rt') as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter="\t", fieldnames=TSV_FIELDNAMES)
        bar = tqdm(reader)
        for item in bar:
            bar.set_description(f'{file_name} iter feature')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default = './')
    args = parser.parse_args()
    file_path = f"{args.path}/ResNet-152-imagenet_80_16_mean"
    mergeFeatures(file_path)
    read_feature(file_path)