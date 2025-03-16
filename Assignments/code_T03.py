import subprocess
import re
import matplotlib.pyplot as plt

def run_program(threads):
    env = {"OMP_NUM_THREADS": str(threads)}
    result = subprocess.run(
        "./a.out", shell=True, capture_output=True, text=True, env=env
    )
    
    Adder_match = re.search(r"Adder Time Taken:\s([\d.]+)", result.stdout)
    Multiplier_match = re.search(r"Multiplier Time Taken:\s([\d.]+)", result.stdout)
    
    Adder_time = float(Adder_match.group(1)) if Adder_match else None
    Multiplier_time = float(Multiplier_match.group(1)) if Multiplier_match else None

    
    return Adder_time, Multiplier_time

THREADS = [1, 2, 4, 6, 8, 10, 12, 16, 20, 32, 64]
Adder = []
Multiplier = []

for thread in THREADS:
    print(f"Running with {thread} threads...")
    Adder1, Multiplier1 = run_program(thread)
    Adder.append(Adder1)
    Multiplier.append(Multiplier1)

print("Threads:", THREADS)
print("Adder Times:", Adder)
print("Multiplier Times:", Multiplier)

def compute_speedup_and_efficiency(times):
    T1 = times[0]
    speedup = [T1 / t  for t in times]
    efficiency = [s / p for s, p in zip(speedup, THREADS)]
    return speedup, efficiency

Adder_speedup, Adder_efficiency = compute_speedup_and_efficiency(Adder)
Multiplier_speedup, Multiplier_efficiency = compute_speedup_and_efficiency(Multiplier)

def compute_parallelization(speedup):
    Fraction = [(1 / s - 1) / (1 / p - 1) for s,p in zip(speedup[1:],THREADS[1:])]
    return Fraction

adder_parallelization_Fraction = compute_parallelization(Adder_speedup)
multiplier_parallelization_Fraction = compute_parallelization(Multiplier_speedup)

print("Adder Speedup:", Adder_speedup)
print("Multiplier Speedup:", Multiplier_speedup)
print("Adder Parallelization Fraction :",adder_parallelization_Fraction)
print("Multiplier Parallelization Fraction :",multiplier_parallelization_Fraction)


plt.figure(figsize=(10, 5))
plt.plot(THREADS, Adder, label="Adder Time",marker = "o")
plt.plot(THREADS, Multiplier, label="Multiplier Time",marker = "s")
plt.legend()
plt.xlabel("Number of Threads")
plt.ylabel("Time Taken (seconds)")
plt.title("Performance Scaling with OpenMP Threads")
# plt.show()

# Plot Speedup
plt.figure(figsize=(10, 5))
plt.plot(THREADS, Adder_speedup, label="Adder Speedup", marker="o")
plt.plot(THREADS, Multiplier_speedup, label="Multiplier Speedup", marker="s")
plt.xlabel("Number of Threads")
plt.ylabel("Speedup")
plt.title("Speedup vs Threads")
plt.legend()
plt.grid()

plt.figure(figsize=(10, 5))
plt.plot(THREADS[1:], adder_parallelization_Fraction, label="Adder Parallelization Fraction", marker="o")
plt.plot(THREADS[1:], multiplier_parallelization_Fraction, label="Multiplier Parallelization Fraction", marker="s")
plt.xlabel("Number of Threads")
plt.ylabel("Parallelization Fraction")
plt.title("Parallelization Fraction vs Threads")
plt.legend()
plt.grid()

plt.show()
