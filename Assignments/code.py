import subprocess
import re
import matplotlib.pyplot as plt

def run_program(threads):
    env = {"OMP_NUM_THREADS": str(threads)}
    result = subprocess.run(
        "./a.out", shell=True, capture_output=True, text=True, env=env
    )
    
    critical_match = re.search(r"Critical Time taken:\s([\d.]+)", result.stdout)
    reduction_match = re.search(r"Reduction Time taken:\s([\d.]+)", result.stdout)
    
    critical_time = float(critical_match.group(1)) if critical_match else None
    reduction_time = float(reduction_match.group(1)) if reduction_match else None
    
    return critical_time, reduction_time

THREADS = [1, 2, 4, 6, 8, 10, 12, 16, 20, 32, 64]
CRITICAL = []
REDUCTION = []

for thread in THREADS:
    print(f"Running with {thread} threads...")
    critical, reduction = run_program(thread)
    CRITICAL.append(critical)
    REDUCTION.append(reduction)

print("Threads:", THREADS)
print("Critical Times:", CRITICAL)
print("Reduction Times:", REDUCTION)

def compute_speedup_and_efficiency(times):
    T1 = times[0]
    speedup = [T1 / t  for t in times]
    efficiency = [s / p for s, p in zip(speedup, THREADS)]
    return speedup, efficiency

critical_speedup, critical_efficiency = compute_speedup_and_efficiency(CRITICAL)
reduction_speedup, reduction_efficiency = compute_speedup_and_efficiency(REDUCTION)

def compute_parallelization(speedup):
    Fraction = [(1 / s - 1) / (1 / p - 1) for s,p in zip(speedup[1:],THREADS[1:])]
    return Fraction

critical_parallelization_Fraction = compute_parallelization(critical_speedup)
reduction_parallelization_Fraction = compute_parallelization(reduction_speedup)

print("Critical Speedup:\n", critical_speedup)
print("Critical Parallelization Fraction:\n", critical_parallelization_Fraction)
print("Reduction Speedup:\n", reduction_speedup)
print("Reduction Parallelization Fraction:\n", reduction_parallelization_Fraction)

plt.figure(figsize=(10,5))
plt.plot(THREADS, CRITICAL, label="Critical Time",marker="o")
plt.plot(THREADS, REDUCTION, label="Reduction Time",marker = "s")
plt.legend()
plt.xlabel("Number of Threads")
plt.ylabel("Time Taken (seconds)")
plt.title("Performance Scaling with OpenMP Threads")
# plt.show()

# Plot Speedup
plt.figure(figsize=(10, 5))
plt.plot(THREADS, critical_speedup, label="Critical Speedup", marker="o")
plt.plot(THREADS, reduction_speedup, label="Reduction Speedup", marker="s")
plt.xlabel("Number of Threads")
plt.ylabel("Speedup")
plt.title("Speedup vs Threads")
plt.legend()
plt.grid()

# plt.show()

# Plot Parallelization
plt.figure(figsize=(10, 5))
plt.plot(THREADS[1:], critical_parallelization_Fraction, label="Critical Parallelization Fraction", marker="o")
plt.plot(THREADS[1:], reduction_parallelization_Fraction, label="Reduction Parallelization Fraction", marker="s")
plt.xlabel("Number of Threads")
plt.ylabel("Parallelization Fraction")
plt.title("Parallelization Fraction vs Threads")
plt.legend()
plt.grid()
plt.show()
