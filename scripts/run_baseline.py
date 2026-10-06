#!/usr/bin/env python3
"""
Baseline Evaluation Runner - Indian Sign Language Translator
Reproduces and verifies baseline metrics against configs/baseline.yaml.
"""
import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_BIN = os.path.join(BASE_DIR, '.venv', 'bin', 'python')
if not os.path.exists(PYTHON_BIN):
    PYTHON_BIN = sys.executable

def main():
    print(f"[*] Running baseline evaluation using interpreter: {PYTHON_BIN}")
    eval_script = os.path.join(BASE_DIR, 'evaluate_all_phases.py')
    cmd = [PYTHON_BIN, eval_script]
    result = subprocess.run(cmd, cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"[!] Baseline evaluation failed with exit code: {result.returncode}")
        sys.exit(result.returncode)
    print("[+] Baseline evaluation completed successfully.")

if __name__ == '__main__':
    main()
