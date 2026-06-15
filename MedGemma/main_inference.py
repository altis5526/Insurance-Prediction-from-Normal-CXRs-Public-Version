import argparse
import subprocess


if __name__ == "__main__":
    subprocess.run(f'python inference.py --check_type train --output_name response_Gemma1.5', shell=True)
    subprocess.run(f'python inference.py --check_type val --output_name response_Gemma1.5', shell=True)
    subprocess.run(f'python inference.py --check_type test --output_name response_Gemma1.5', shell=True)