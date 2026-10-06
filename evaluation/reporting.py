"""
Reporting and Serialization Utility
"""
import json
import csv

def save_json_report(data, filepath):
    # Convert numpy types to native python
    def default_converter(o):
        if hasattr(o, 'item'):
            return o.item()
        if hasattr(o, 'tolist'):
            return o.tolist()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")
        
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=default_converter)

def save_csv_report(rows, filepath, fieldnames=None):
    if not rows:
        return
    if fieldnames is None:
        if isinstance(rows[0], dict):
            fieldnames = list(rows[0].keys())
    with open(filepath, 'w', newline='') as f:
        if isinstance(rows[0], dict):
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        else:
            writer = csv.writer(f)
            writer.writerows(rows)
