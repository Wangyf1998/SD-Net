import os
import pandas as pd
csv_path = "/home/wyf/final_combined.csv"
npz_path = "/home/wyf/open_source_dataset/artemis_dataset/wikiart_CLIP_101_49/"
a = pd.read_csv(csv_path, usecols='painting').tolist()
for i in a:
    filename = os.path.join(a, ".npz").replace('/', '')
    npz = os.path.join(npz_path, filename)
    if not os.path.exists(npz):
        print(filename)
        continue




