import random

from multiprocessing import Pool, cpu_count

import numpy as np


def element(index, A, B):
    i, j = index
    res = 0
    N = len(A[0]) or len(B)
    for k in range(N):
        res += A[i][k] * B[k][j]
    return res


def parallel_multiply_matrices(A, B):
    if len(A) == 0 or len(B) == 0:
        raise ValueError("Matrices cannot have zero dimensions")
    if len(A[0]) != len(B):
        raise ValueError("The number of columns in matrix A must be equal to the number of rows in matrix B")
    if len(A) != len(A[0]):
        raise ValueError("Matrix A must be square")
    if len(B) != len(B[0]):
        raise ValueError("Matrix B must be square")

    indices = [(i, j) for i in range(len(A)) for j in range(len(B[0]))]

    with Pool(cpu_count()) as pool:
        result_elements = pool.starmap(element, [(index, A, B) for index in indices])

    result_matrix = np.reshape(result_elements, (len(A), len(B[0])))

    return result_matrix


if __name__ == '__main__':
    while True:
        try:
            a, b = map(int, input("Enter dimensions of matrix A (a, b): ").split())
            c, d = map(int, input("Enter dimensions of matrix B (c, d): ").split())
            if a <= 0 or b <= 0 or c <= 0 or d <= 0:
                raise ValueError("Dimensions must be positive integers")
            break
        except ValueError as e:
            print(e)

    A = np.random.randint(0, 10, size=(a, a))
    B = np.random.randint(0, 10, size=(b, b))

    try:
        result = parallel_multiply_matrices(A, B)

        print("Matrix A:")
        print(A)
        print("\nMatrix B:")
        print(B)
        print("\nResult:")
        print(result)

    except ValueError as e:
        print(e)