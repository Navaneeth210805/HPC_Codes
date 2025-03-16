import matplotlib.pyplot as plt
import numpy as np

# Define the thread counts used in C program
THREADS = [1, 2, 4, 6, 8, 10, 12, 16, 20, 32, 64]

# Read execution times from the file
def read_execution_times(filename):
    with open(filename, "r") as file:
        times = [float(line.strip()) for line in file]
    return times

# Load data
filename = "benchmark_results.txt"
times = read_execution_times(filename)

print(len(times))
# Calculate speedup
T1 = times[0]  # Time taken with 1 thread
speedup = [T1 / t for t in times]

# Calculate parallelization factor using the given formula
parallel_factor = [((1 / s) - 1) / ((1 / p) - 1) for s, p in zip(speedup[1:], THREADS[1:])]
print(*speedup)
print(*parallel_factor)

# # Plot execution time
# plt.figure(figsize=(12, 5))

# plt.plot(THREADS, times, marker='o', linestyle='-', color='b', label='Execution Time')
# plt.xlabel("Number of Threads")
# plt.ylabel("Time Taken (seconds)")
# plt.title("Parallel Performance")
# plt.xticks(THREADS)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend()
# plt.show()

# # Plot speedup
# plt.plot(THREADS, speedup, marker='s', linestyle='-', color='g', label='Speedup')
# plt.xlabel("Number of Threads")
# plt.ylabel("Speedup")
# plt.title("Speedup vs Threads")
# plt.xticks(THREADS)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend()
# plt.show()

# # Plot parallelization factor

# plt.plot(THREADS[1:], parallel_factor, marker='^', linestyle='-', color='r', label='Parallelization Factor')
# plt.xlabel("Number of Threads")
# plt.ylabel("Parallelization Factor")
# plt.title("Parallelization Factor")
# plt.xticks(THREADS)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend()
# plt.show()
