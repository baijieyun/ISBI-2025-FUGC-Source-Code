#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   metrics_logger.py
@Time    :   2025/01/20 01:38:57
@Author  :   biabuluo 
@Version :   1.0
@Desc    :   记录每个epoch指标
'''
import csv
import os

class MetricsLogger:
    def __init__(self, save_dir, file_name="metrics.csv"):
        self.save_dir = save_dir
        self.file_name = file_name
        self.file_path = os.path.join(save_dir, file_name)
        self.fields = ["epoch", "train_loss", 'hd', 'hd_upper', 'hd_lower', 'hd_all', 'dsc', 'dsc_1', 'dsc_2']
        
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Create the CSV file and write the header
        with open(self.file_path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writeheader()

    def log_metrics(self, epoch, train_loss, hd, hd_upper, hd_lower, hd_all, dsc, dsc_1, dsc_2):
        """Log metrics for a specific epoch."""
        print({
                "epoch": epoch,
                "train_loss": train_loss,
                "hd": hd,
                "hd_upper": hd_upper,
                "hd_lower": hd_lower,
                "hd_all": hd_all,
                "dsc": dsc,
                "dsc_1": dsc_1,
                "dsc_2": dsc_2
            })
        with open(self.file_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow({
                "epoch": epoch,
                "train_loss": train_loss,
                "hd": hd,
                "hd_upper": hd_upper,
                "hd_lower": hd_lower,
                "hd_all": hd_all,
                "dsc": dsc,
                "dsc_1": dsc_1,
                "dsc_2": dsc_2
            })

    def get_metrics(self):
        """Load all logged metrics as a list of dictionaries."""
        with open(self.file_path, mode="r") as f:
            reader = csv.DictReader(f)
            return list(reader)

# Example usage
if __name__ == "__main__":
    logger = MetricsLogger(save_dir="outputs/logs")
    logger.log_metrics(epoch=1, train_loss=0.1, val_dsc=0.8, val_hd=15, params=1.2e6, flops=3.4e9, inference_speed=0.02)
    print("Logged Metrics:")
    print(logger.get_metrics())

