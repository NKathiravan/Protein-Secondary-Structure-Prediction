import subprocess
from datetime import datetime

# List of files to run
files = [
    "QuantumAngularEncoding.py",
    "QuantumFeedforward.py",
    "QuantumReencoding.py",
    "QuantumSelfAttention.py"
]

# Output log file
log_file = "quantum_outputs.log"

with open(log_file, "w") as f:
    for file in files:
        f.write("\n\n===========================\n")
        f.write(f"Running: {file}\n")
        f.write(f"Started at: {datetime.now()}\n")
        f.write("===========================\n\n")
        f.flush()

        try:
            process = subprocess.Popen(
                ["python", file],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            # Stream output line-by-line
            for line in process.stdout:
                print(line, end="")       # Show on console
                f.write(line)             # Save to file
                f.flush()

            process.wait()

        except Exception as e:
            f.write(f"\n[ERROR] Failed to run {file}: {str(e)}\n")

        f.write(f"\nCompleted: {file} at {datetime.now()}\n")
        f.write("---------------------------\n")
        f.flush()

print(f"\nAll scripts executed. Logs saved to {log_file}")
