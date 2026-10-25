#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_script(script_name: str) -> None:
    script_path = ROOT / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    print(f"\n=== Running {script_name} ===")
    subprocess.run([sys.executable, str(script_path)], check=True, cwd=str(ROOT))


if __name__ == "__main__":
    try:
        run_script("prdGenerator.py")
        run_script("requirementGatheringAgent.py")
        run_script("testCaseGenerator.py")
        print("\n✅ Full PRD-to-requirements-to-test-case pipeline completed successfully.")
    except subprocess.CalledProcessError as exc:
        print(f"\n❌ Pipeline failed with exit code {exc.returncode}.")
        raise SystemExit(exc.returncode)
