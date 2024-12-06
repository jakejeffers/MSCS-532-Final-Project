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

# Generating random matrices
matrix_size = 100
A = np.random.rand(matrix_size, matrix_size).tolist()
B = np.random.rand(matrix_size, matrix_size).tolist()

# Timing the unoptimized approach
start_time = time.time()
unoptimized_result = unoptimized_matrix_multiply(A, B)
unoptimized_time = time.time() - start_time

# Convert lists to NumPy arrays for optimized approach
A_np = np.array(A)
B_np = np.array(B)

# Timing the optimized approach
start_time = time.time()
optimized_result = np.dot(A_np, B_np)
optimized_time = time.time() - start_time

print(f"Unoptimized Time for {matrix_size}x{matrix_size}: {unoptimized_time:.5f} seconds")
print(f"Optimized Time for {matrix_size}x{matrix_size}: {optimized_time:.5f} seconds")

