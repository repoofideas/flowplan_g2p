#!/usr/bin/env python3
import json
from collections import defaultdict

import sys

files = sys.argv[1:] if len(sys.argv) > 1 else []
if not files:
    print("Usage: python -m flowplan_g2p.evaluation.calculate_average_scores <file1.json> [file2.json ...]")
    sys.exit(1)

for file in files:
    totals = defaultdict(lambda: defaultdict(list))
    
    with open(file) as f:
        data = json.load(f)
        for result in data["results"]:
            for metric, scores in result.items():
                if metric != "doc_id" and isinstance(scores, dict):
                    for score_type, value in scores.items():
                        totals[metric][score_type].append(value)
    
    print(f"{file.replace('_evaluation_results.json', '').upper()}:")
    print("=" * 40)
    for metric in totals:
        print(f"\n{metric.upper()}:")
        for score_type, values in totals[metric].items():
            avg = sum(values) / len(values)
            print(f"  {score_type}: {avg:.4f}")
    print("\n")
