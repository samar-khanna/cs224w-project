import os
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to tabulate results")
    parser.add_argument('--path', default=None, help="location of log.txt")
    args = parser.parse_args()

    val_tp=0.0
    val_fn=0.0
    val_tn=0.0
    val_fp=0.0
    test_tp=0.0
    test_fn=0.0
    test_tn=0.0
    test_fp=0.0
    with open(args.path,"r") as f:
        lines = f.readlines()
        for line in lines:
            if line[:5] =="VAL t":
                s = line.split(':')
                val_tp += float(s[1].split(',')[0][1:])
                val_fn += float(s[2].split(',')[0][1:])
                val_tn += float(s[3].split(',')[0][1:])
                val_fp += float(s[4].split(',')[0][1:])
            elif line[:6] =="TEST t":
                s = line.split(':')
                test_tp += float(s[1].split(',')[0][1:])
                test_fn += float(s[2].split(',')[0][1:])
                test_tn += float(s[3].split(',')[0][1:])
                test_fp += float(s[4].split(',')[0][1:])
    
    print(f"VAL tp:{val_tp}, fn:{val_fn}, tn:{val_tn}, fp:{val_fp}")
    print(f"TEST tp:{test_tp}, fn:{test_fn}, tn:{test_tn}, fp:{test_fp}")