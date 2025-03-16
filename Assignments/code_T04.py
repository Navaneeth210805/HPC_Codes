import subprocess
import re
import matplotlib.pyplot as plt

def run_program(threads):
    env = {"OMP_NUM_THREADS": str(threads)}
    result = subprocess.run(
        "./a.out", shell=True, capture_output=True, text=True, env=env
    )
    print(result.stdout)
    
    Multiplication_match = re.search(r"Multiplication Time:\s([\d.]+)", result.stdout)
    sum_match = re.search(r"Sum Time:\s([\d.]+)", result.stdout)
    
    Multiplication_time = float(Multiplication_match.group(1)) if Multiplication_match else None
    sum_time = float(sum_match.group(1)) if sum_match else None

    
    return Multiplication_time, sum_time

THREADS = [1, 2, 4, 6, 8, 10, 12, 16, 20, 32, 64]
sum = []
Multiplier = []

for thread in THREADS:
    print(f"Running with {thread} threads...")
    Multiplication_time, sum_time = run_program(thread)
    sum.append(sum_time)
    Multiplier.append(Multiplication_time)

print("Threads:", THREADS)
print("Sum Time:", sum)
print("Multiplication Time:", Multiplier)

def compute_speedup_and_efficiency(times):
    T1 = times[0]
    speedup = [T1 / t  for t in times]
    efficiency = [s / p for s, p in zip(speedup, THREADS)]
    return speedup, efficiency

sum_speedup, sum_efficiency = compute_speedup_and_efficiency(sum)
Multiplier_speedup, Multiplier_efficiency = compute_speedup_and_efficiency(Multiplier)

def compute_parallelization(speedup):
    Fraction = [(1 / s - 1) / (1 / p - 1) for s,p in zip(speedup[1:],THREADS[1:])]
    return Fraction

adder_parallelization_Fraction = compute_parallelization(sum_speedup)
multiplier_parallelization_Fraction = compute_parallelization(Multiplier_speedup)

print("Sum Speedup:", sum_speedup)
print("Multiplication Speedup:", Multiplier_speedup)
print("Sum Parallelization Fraction :",adder_parallelization_Fraction)
print("Multiplication Parallelization Fraction :",multiplier_parallelization_Fraction)


plt.figure(figsize=(10, 5))
plt.plot(THREADS, sum, label="Sum Time",marker = "o")
plt.plot(THREADS, Multiplier, label="Multiplication Time",marker = "s")
plt.legend()
plt.xlabel("Number of Threads")
plt.ylabel("Time Taken (seconds)")
plt.title("Performance Scaling with OpenMP Threads")
# plt.show()

# Plot Speedup
plt.figure(figsize=(10, 5))
plt.plot(THREADS, sum_speedup, label="Sum Speedup", marker="o")
plt.plot(THREADS, Multiplier_speedup, label="Multiplication Speedup", marker="s")
plt.xlabel("Number of Threads")
plt.ylabel("Speedup")
plt.title("Speedup vs Threads")
plt.legend()
plt.grid()

plt.figure(figsize=(10, 5))
plt.plot(THREADS[1:], adder_parallelization_Fraction, label="Sum Parallelization Fraction", marker="o")
plt.plot(THREADS[1:], multiplier_parallelization_Fraction, label="Multiplication Parallelization Fraction", marker="s")
plt.xlabel("Number of Threads")
plt.ylabel("Parallelization Fraction")
plt.title("Parallelization Fraction vs Threads")
plt.legend()
plt.grid()

plt.show()
