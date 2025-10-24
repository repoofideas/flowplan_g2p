#!/usr/bin/env python3
import json
from collections import defaultdict

files = [
    "sonnet4.5_evaluation_results.json",

]

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
