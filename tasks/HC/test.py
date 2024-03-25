from utils import load_datasets, load_nav_graphs, relHumanAngle
splits = ['train']
for item in load_datasets(splits)[:2]:
    print(item['scan'])