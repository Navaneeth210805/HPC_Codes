import subprocess
import matplotlib.pyplot as plt

subprocess.run(['gcc', '-fopenmp', 'matrix_addition.c', '-o', 'matrix_addition'], check=True)

threads = [1, 2, 4, 6, 8, 10, 12, 16, 20, 32, 64]
parallel_times = []
for t in threads:
    print(f"Running with {t} threads...")
    result = subprocess.run(['./matrix_addition', str(t)], capture_output=True, text=True)
    output = result.stdout.strip().split('\n')

    parallel_time = None
    for line in output:
        if "Parallel Matrix Addition Time" in line:
            parallel_time = float(line.split()[-2])
            break

    if parallel_time is None:
        print(f"Error: Could not extract execution time for {t} threads.")
        exit(1)

    parallel_times.append(parallel_time)

print("\nExecution Time Table")
print("=" * 50)
print(f"{'Threads':<10} {'Execution Time (s)':<20}")
print("-" * 50)
for t, time in zip(threads, parallel_times):
    print(f"{t:<10} {time:<20.6f}")
print("=" * 50)

speedup = [parallel_times[0] / t for t in parallel_times]

plt.figure(figsize=(8, 6))
plt.plot(threads, parallel_times, 'bo-', label='Parallel Execution Time')
plt.xlabel('Number of Threads')
plt.ylabel('Execution Time (s)')
plt.title('Execution Time vs Threads')
plt.legend()
plt.grid()
plt.savefig('execution_time.png')
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(threads, speedup, 'go-', label='Speedup')
plt.xlabel('Number of Threads')
plt.ylabel('Speedup')
plt.title('Speedup vs Threads')
plt.legend()
plt.grid()
plt.savefig('speedup.png')
plt.close()

# def estimate_p(S, p):
#     if p == 1 or S == 0:
#         return 0.0  # Avoid division by zero
#     P = (S - 1) / (S * (1 - 1/p))
#     return max(0.0, min(P, 1.0))  # Ensure P is between 0 and 1

# # Calculate parallelization fractions for all thread counts
# parallel_fractions = [estimate_p(speedup[i], threads[i]) for i in range(len(threads))]

# # Print parallelization fractions in a table
# print("\nParallelization Fraction Table")
# print("=" * 50)
# print(f"{'Threads':<10} {'Parallelization (f)':<15}")
# print("-" * 50)
# for t, f in zip(threads, parallel_fractions):
#     print(f"{t:<10} {f:<15.4f}")
# print("=" * 50)
