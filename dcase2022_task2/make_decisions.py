from pathlib import Path
import csv

import numpy as np

from .datasets import *
from .utils.measures import *

__all__ = ["find_threshold"]

def make_decisions(args, system):
    print("-------------------------")
    print("|MAKE DECISIONS SYSTEM {}|".format(system))
    print("-------------------------")

    #======Read args========================================================= 
    path = Path(__file__).resolve().parents[1]
    csv_dir = path/"results"/args.config_data/args.config_class/args.dataset/\
        ("Almudevar_UZ_task2_{}".format(system))
    Path(csv_dir).mkdir(parents=True, exist_ok=True)

    sections = EVAL_SECTIONS if args.dataset=="eval" else DEV_SECTIONS
    for machine in MACHINES:
        for section in sections:
            csv_file = "anomaly_score_{}_section_{:02d}_test.csv".format(machine, section)    
            with open(csv_dir/csv_file, 'r') as f:
                reader = csv.reader(f)
                results = {row[0]: float(row[1]) for row in reader}
                
            threshold = np.median(list(results.values()))
            decide = lambda cd: 1 if cd>threshold else 0
            decision = {file: decide(results[file]) for file in results}

            csv_file = "decision_result_{}_section_{:02d}_test.csv".format(machine, section)
            with open(csv_dir/csv_file, 'w') as f:
                for key in decision.keys():
                    f.write("%s,%s\n" % (key, decision[key]))