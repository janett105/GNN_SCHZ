import csv
import json

csvfile = 'COBRE_parameters_rest.csv'
jsonfile = 'COBRE_parameters_rest.json'

with open(csvfile, "r", encoding="utf-8", newline="") as input_file, \
open(jsonfile, "w", encoding=" utf-8", newline="") as output_file:
    reader = csv.reader(input_file)

    doc={}
    for (col_name, col) in reader:
        doc[col_name] = col
    print(doc)
    print(json.dumps(doc, ensure_ascii=False), file=output_file)