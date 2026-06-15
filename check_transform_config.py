#!/usr/bin/env python3
"""
Script to check all training scripts for proper dataset transform configuration.
When mode == "test", train_dataset should have transform=True and val_dataset should have transform=False.
"""

import os
import re

# Get all training script files
script_dir = "/home/sebasmos/orcd/pool/code/JAMA_codes"
train_scripts = [f for f in os.listdir(script_dir) if f.startswith('train_') and f.endswith('.py') and not f.startswith('.')]

print("=" * 80)
print("CHECKING DATASET TRANSFORM CONFIGURATION IN TRAINING SCRIPTS")
print("=" * 80)
print()

issues_found = []

for script in sorted(train_scripts):
    script_path = os.path.join(script_dir, script)

    with open(script_path, 'r') as f:
        content = f.read()
        lines = content.split('\n')

    # Find the test mode section
    test_mode_found = False
    test_section_lines = []
    in_test_section = False

    for i, line in enumerate(lines):
        if 'elif args.mode == "test"' in line:
            test_mode_found = True
            in_test_section = True
            start_idx = i

        if in_test_section:
            test_section_lines.append((i+1, line))  # line numbers are 1-indexed

            # Stop collecting when we hit the next major section (like epochs assignment)
            if i > start_idx and (line.strip().startswith('epochs =') or
                                   line.strip().startswith('demo_labels_dict =') or
                                   line.strip().startswith('if args.patch_idx')):
                break

    # Search for dataset creation patterns
    train_dataset_pattern = r'train_dataset\s*=\s*\w+\([^)]*transform\s*=\s*(True|False|training)[^)]*\)'
    val_dataset_pattern = r'val_dataset\s*=\s*\w+\([^)]*transform\s*=\s*(True|False|training)[^)]*\)'

    train_dataset_no_transform = r'train_dataset\s*=\s*\w+\([^)]*\)(?!.*transform)'
    val_dataset_no_transform = r'val_dataset\s*=\s*\w+\([^)]*\)(?!.*transform)'

    # Search in the whole file for dataset creation
    train_transform_match = re.search(train_dataset_pattern, content)
    val_transform_match = re.search(val_dataset_pattern, content)

    train_no_transform = re.search(train_dataset_no_transform, content)
    val_no_transform = re.search(val_dataset_no_transform, content)

    has_issues = False
    issue_details = []

    if test_mode_found:
        # Check if datasets specify transform parameter
        if train_no_transform and not train_transform_match:
            has_issues = True
            issue_details.append("  ❌ train_dataset does NOT specify transform parameter")
        elif train_transform_match:
            transform_val = train_transform_match.group(1)
            if transform_val == "True":
                issue_details.append("  ✓ train_dataset has transform=True")
            elif transform_val == "False":
                has_issues = True
                issue_details.append("  ❌ train_dataset has transform=False (should be True)")
            elif transform_val == "training":
                issue_details.append("  ⚠ train_dataset uses transform=training (conditional)")

        if val_no_transform and not val_transform_match:
            has_issues = True
            issue_details.append("  ❌ val_dataset does NOT specify transform parameter")
        elif val_transform_match:
            transform_val = val_transform_match.group(1)
            if transform_val == "False":
                issue_details.append("  ✓ val_dataset has transform=False")
            elif transform_val == "True":
                has_issues = True
                issue_details.append("  ❌ val_dataset has transform=True (should be False for test)")
            elif transform_val == "training":
                issue_details.append("  ⚠ val_dataset uses transform=training (conditional)")
    else:
        has_issues = True
        issue_details.append("  ❌ No test mode section found")

    if has_issues or issue_details:
        issues_found.append(script)
        print(f"📄 {script}")
        for detail in issue_details:
            print(detail)
        print()

print("=" * 80)
print(f"SUMMARY: {len(issues_found)} files with issues or warnings out of {len(train_scripts)} total")
print("=" * 80)
if issues_found:
    print("\nFiles needing attention:")
    for script in issues_found:
        print(f"  - {script}")
