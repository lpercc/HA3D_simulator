import numpy as np
import csv
import base64
def read_tsv(tsv_path, TSV_FIELDNAMES, WIDTH, HEIGHT):
    # Verify we can read a tsv
    in_data = []
    with open(tsv_path, "rt") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter="\t", fieldnames=TSV_FIELDNAMES)
        for item in reader:
            item["scanId"] = item["scanId"]
            item["step"] = int(item["step"])
            item["rgb"] = np.frombuffer(
                base64.b64decode(item["rgb"]), dtype=np.uint8
            ).reshape((HEIGHT, WIDTH, 3))
            item["depth"] = np.frombuffer(
                base64.b64decode(item["depth"]), dtype=np.uint16
            ).reshape((HEIGHT, WIDTH))
            item["location"] = item["location"]
            item["heading"] = float(item["heading"])
            item["elevation"] = float(item["elevation"])
            item["viewIndex"] = int(item["viewIndex"])
            item["navigableLocations"] = item["navigableLocations"]
            in_data.append(item)
    return in_data