#!/usr/bin/env python3
"""
Experiment Status Checker
=========================
This script checks the status of all training experiments by reading the STATUS.txt files
created by the training scripts.

Usage:
    python check_experiment_status.py

    or with specific directories:
    python check_experiment_status.py --weight_dir sexage_sexagerace_weights
"""

import os
import glob
import argparse
from datetime import datetime
from pathlib import Path

def read_status_file(filepath):
    """Read and parse a status file."""
    if not os.path.exists(filepath):
        return None

    status_info = {}
    with open(filepath, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                status_info[key.strip()] = value.strip()
    return status_info

def format_status_display(status_info, filepath):
    """Format the status information for display."""
    if not status_info:
        return None

    status = status_info.get('STATUS', 'UNKNOWN')
    model = status_info.get('MODEL', 'Unknown')
    experiment = status_info.get('EXPERIMENT', 'Unknown')

    # Color coding based on status
    if status == 'COMPLETED' or status == 'STOPPED_EARLY':
        status_symbol = '✓'
        status_color = '\033[92m'  # Green
    elif status == 'RUNNING':
        status_symbol = '▶'
        status_color = '\033[93m'  # Yellow
    elif status == 'FAILED':
        status_symbol = '✗'
        status_color = '\033[91m'  # Red
    else:
        status_symbol = '?'
        status_color = '\033[90m'  # Gray

    reset_color = '\033[0m'
    bold = '\033[1m'

    # Build the output
    output = []
    output.append(f"\n{status_color}{'='*70}{reset_color}")
    output.append(f"{status_color}{status_symbol} {status}{reset_color} - {model} - {experiment}")
    output.append(f"{status_color}{'='*70}{reset_color}")

    if status == 'RUNNING':
        current_epoch = status_info.get('CURRENT_EPOCH', 'Unknown')
        last_auc = status_info.get('LAST_AUC', 'N/A')
        last_acc = status_info.get('LAST_ACC', 'N/A')
        best_auc = status_info.get('BEST_AUC', status_info.get('MAX_AUC', 'N/A'))
        best_acc = status_info.get('BEST_ACC', status_info.get('MAX_ACC', 'N/A'))
        start_time = status_info.get('START_TIME', 'Unknown')

        output.append(f"  Progress: {current_epoch}")
        output.append(f"  Started: {start_time}")
        output.append(f"  Last Validation AUC: {last_auc} | Best Validation AUC: {best_auc}")
        output.append(f"  Last Validation ACC: {last_acc} | Best Validation ACC: {best_acc}")

    elif status == 'COMPLETED' or status == 'STOPPED_EARLY':
        completed_epochs = status_info.get('COMPLETED_EPOCHS', 'Unknown')
        total_epochs = status_info.get('TOTAL_EPOCHS', status_info.get('TOTAL_EPOCHS_PLANNED', 'Unknown'))
        final_auc = status_info.get('FINAL_AUC', 'N/A')
        final_acc = status_info.get('FINAL_ACC', 'N/A')
        best_auc = status_info.get('BEST_AUC', status_info.get('MAX_AUC', 'N/A'))
        best_acc = status_info.get('BEST_ACC', status_info.get('MAX_ACC', 'N/A'))
        end_time = status_info.get('END_TIME', 'Unknown')

        # Check for test results
        paper_auc = status_info.get('PAPER_REPORTABLE_AUC', None)
        test_auc = status_info.get('TEST_AUC', 'N/A')
        test_precision = status_info.get('TEST_PRECISION', 'N/A')
        test_recall = status_info.get('TEST_RECALL', 'N/A')
        test_f1 = status_info.get('TEST_F1', 'N/A')
        test_acc = status_info.get('TEST_ACC', 'N/A')
        test_time = status_info.get('TEST_TIME', None)

        if status == 'STOPPED_EARLY':
            reason = status_info.get('REASON', 'Unknown')
            output.append(f"  Stopped: {reason}")
            output.append(f"  Completed: {completed_epochs}/{total_epochs} epochs")
        else:
            output.append(f"  Completed: {completed_epochs}/{total_epochs} epochs")
        output.append(f"  Finished: {end_time}")
        output.append(f"  Training - Best Validation AUC: {best_auc} | Best Validation ACC: {best_acc}")

        # Highlight test results if available
        if paper_auc or test_auc != 'N/A':
            output.append(f"")
            output.append(f"{bold}  *** FINAL TEST RESULTS (HELD-OUT TEST SET) ***{reset_color}")
            if paper_auc:
                output.append(f"{bold}  📊 PAPER-REPORTABLE AUC: {paper_auc}{reset_color}")
            output.append(f"  Test AUC: {test_auc}")
            output.append(f"  Test Precision: {test_precision} | Recall: {test_recall} | F1: {test_f1}")
            output.append(f"  Test Accuracy: {test_acc}")
            if test_time:
                output.append(f"  Tested: {test_time}")

    elif status == 'FAILED':
        error = status_info.get('ERROR', 'Unknown error')
        error_time = status_info.get('ERROR_TIME', 'Unknown')

        output.append(f"  Failed at: {error_time}")
        output.append(f"  Error: {error}")

    output.append(f"  File: {filepath}")

    return '\n'.join(output)

def check_experiments(base_dir='.'):
    """Check all experiment status files in the given directory."""

    # Find all STATUS.txt files
    status_files = glob.glob(f"{base_dir}/**/*_STATUS.txt", recursive=True)

    if not status_files:
        print(f"\n{' '*10}No experiment status files found!")
        print(f"{' '*10}STATUS.txt files will be created when you run training scripts.")
        return

    # Categorize by status
    completed = []
    running = []
    failed = []
    unknown = []

    for filepath in sorted(status_files):
        status_info = read_status_file(filepath)
        if status_info:
            status = status_info.get('STATUS', 'UNKNOWN')
            if status == 'COMPLETED' or status == 'STOPPED_EARLY':
                completed.append((filepath, status_info))
            elif status == 'RUNNING':
                running.append((filepath, status_info))
            elif status == 'FAILED':
                failed.append((filepath, status_info))
            else:
                unknown.append((filepath, status_info))

    # Print summary
    total = len(status_files)
    print(f"\n{'='*70}")
    print(f"EXPERIMENT STATUS SUMMARY")
    print(f"{'='*70}")
    print(f"  Total experiments: {total}")
    print(f"  ✓ Completed: {len(completed)}")
    print(f"  ▶ Running: {len(running)}")
    print(f"  ✗ Failed: {len(failed)}")
    if unknown:
        print(f"  ? Unknown: {len(unknown)}")
    print(f"{'='*70}\n")

    # Display running experiments first (most important)
    if running:
        print("\n" + "="*70)
        print("CURRENTLY RUNNING EXPERIMENTS")
        print("="*70)
        for filepath, status_info in running:
            output = format_status_display(status_info, filepath)
            if output:
                print(output)

    # Then completed
    if completed:
        print("\n" + "="*70)
        print("COMPLETED EXPERIMENTS")
        print("="*70)
        for filepath, status_info in completed:
            output = format_status_display(status_info, filepath)
            if output:
                print(output)

    # Then failed
    if failed:
        print("\n" + "="*70)
        print("FAILED EXPERIMENTS")
        print("="*70)
        for filepath, status_info in failed:
            output = format_status_display(status_info, filepath)
            if output:
                print(output)

    # Finally unknown
    if unknown:
        print("\n" + "="*70)
        print("UNKNOWN STATUS EXPERIMENTS")
        print("="*70)
        for filepath, status_info in unknown:
            output = format_status_display(status_info, filepath)
            if output:
                print(output)

    print("\n" + "="*70)
    print("NOTE: If experiments show as RUNNING but no process is active,")
    print("      they were likely preempted by the cluster scheduler.")
    print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description='Check the status of all training experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Check all experiments in current directory:
    python check_experiment_status.py

  Check experiments in specific weight directory:
    python check_experiment_status.py --weight_dir sexage_sexagerace_weights

  Check experiments in multiple directories:
    python check_experiment_status.py --weight_dir sexage_weights sexage_sexagerace_weights
        """
    )

    parser.add_argument(
        '--weight_dir',
        nargs='+',
        default=['.'],
        help='Directories to search for status files (default: current directory)'
    )

    args = parser.parse_args()

    # Check each directory
    for weight_dir in args.weight_dir:
        if not os.path.exists(weight_dir):
            print(f"\nWarning: Directory '{weight_dir}' not found, skipping...")
            continue

        if len(args.weight_dir) > 1:
            print(f"\n\n{'#'*70}")
            print(f"# Checking directory: {weight_dir}")
            print(f"{'#'*70}")

        check_experiments(weight_dir)

if __name__ == '__main__':
    main()
