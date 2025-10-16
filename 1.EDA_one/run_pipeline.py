import subprocess
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def run_script(script_name):
    print(f"\n🔹 Running: {script_name}\n{'-'*50}")
    
    process = subprocess.Popen(
        [sys.executable, script_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1  # Line-buffered
    )

    # Stream output line-by-line
    for line in process.stdout:
        print(line, end='')  # Already includes newline

    process.wait()

    if process.returncode != 0:
        raise RuntimeError(f"\n❌ Script {script_name} failed with return code {process.returncode}. Stopping pipeline.")
    
    print(f"\n✅ Done with: {script_name}\n{'='*50}")

try:
    run_script("preprocessing.py")                                
    run_script("optuna_rlstm_train.py")
    run_script("final_rlstm_model.py")
    run_script("ensemble_rlstm_uncertainty.py")
    run_script("predict_ensemble_test.py")
    run_script("evaluate_ensemble_full.py")
    print("\n🎉 All steps completed successfully.")

except RuntimeError as e:
    print(f"{e}")

except Exception as e:
    print(f"❌ Unexpected error: {e}")
