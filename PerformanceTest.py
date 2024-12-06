import time
import numpy as np

# Function for unoptimized matrix multiplication
def unoptimized_matrix_multiply(A, B):
    rows_A = len(A)
    cols_A = len(A[0])
    cols_B = len(B[0])
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    return result

# Benchmarking function
def benchmark_matrix_multiplication(sizes):
    results = []

    for size in sizes:
        # Generate random matrices
        A = np.random.rand(size, size).tolist()
        B = np.random.rand(size, size).tolist()
        A_np = np.array(A)
        B_np = np.array(B)

        # Unoptimized
        start_time = time.time()
        unoptimized_matrix_multiply(A, B)
        unoptimized_time = time.time() - start_time

        # Optimized
        start_time = time.time()
        np.dot(A_np, B_np)
        optimized_time = time.time() - start_time

        # Append results
        results.append((size, unoptimized_time, optimized_time))

    return results

# List of matrix sizes to test
matrix_sizes = [100, 200, 500, 1000]
benchmark_results = benchmark_matrix_multiplication(matrix_sizes)

# Display results
print(f"{'Size':<10}{'Unoptimized Time (s)':<25}{'Optimized Time (s)':<25}")
for size, unoptimized, optimized in benchmark_results:
    print(f"{size:<10}{unoptimized:<25.5f}{optimized:<25.5f}")
