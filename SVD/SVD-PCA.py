# SVD-PCA for CSE 5800
# Written by Joshua Sheldon

# A program which attempts to use the output of Singular
# Value Decomposition to perform Principal Component
# Analysis

# ---------- Imports ----------

import matplotlib.pyplot as plt
import numpy as np

# ---------- Utility Functions ----------


def expand_singular_values_array(S: np.ndarray, M: int, N: int):
    """
    Given a one dimensional array of singular values and the
    dimensions of the matrix from which they were obtained,
    returns a new matrix with the dimensions of the original
    matrix with the singular values placed on the diagonal.

    :param np.ndarray S: The singular values obtained by performing
                         Singular Value Decomposition on the matrix.
    :param int M:        The number of rows of the original matrix.
    :param int N:        The number of columns of the original matrix.
    :return:             An M x N matrix with the values of S
                         placed on the diagonal.
    """

    # Create blank M x N matrix
    new_S = np.zeros((M, N))

    # Place singular values on the diagonal
    for i in range(len(S)):
        new_S[i][i] = S[i]

    # Return singular values matrix
    return new_S


# ---------- DEFINE MATRIX HERE ----------

matrix = np.array([[1, 2, 3, 4, 5],
                   [6, 7, 8, 9, 10],
                   [11, 12, 13, 14, 15],
                   [16, 17, 18, 19, 20],
                   [21, 22, 23, 24, 25]])

# ---------- Business Logic ----------

# Print original matrix to compare with the
# matrix we reconstruct later
print(f"~Original Matrix~\n{matrix}\n")

# Pull matrix height and width from number of
# rows and number of columns within the first
# row
M = len(matrix)
N = len(matrix[0])

# Center the data (required for Principal Component
# Analysis)
centered_matrix = matrix - np.mean(matrix, axis=0)
print(f"~Centered Matrix~\n{centered_matrix}\n")

# Use NumPy to perform Singular Value Decomposition
# on the matrix
svd = np.linalg.svd(centered_matrix)

# Assign SVD output as variables
U = svd.U
S = svd.S
Vh = svd.Vh

# Expands the array of singular values into a matrix with
# the same dimensions of the centered matrix
S_expanded = expand_singular_values_array(S, M, N)

# Print decomposed matrices
print(f"~U (Left Singular Vectors)~\n{U}\n")
print(f"~S (Singular Values)~\n{S}\n")
print(f"~Vh (Principal Components / Right Singular Vectors)~\n{Vh}\n")

# Prove SVD was successful by reconstructing the
# original matrix from the decomposed matrices
print(f"~Reconstructed Matrix~\n{U @ S_expanded @ Vh}\n")

# Begin PCA
S_squared = S**2
explained_variance = S_squared / np.sum(S_squared)
print(f"~Explained Variance by Each Component~\n{explained_variance}\n")

# Graph explained variance to educate the user's choice
# of number of components for PCA
x_coords = [i for i in range(len(explained_variance))]
plt.plot(x_coords, explained_variance)
plt.xlabel("Component Index")
plt.ylabel("Explained Variance")
plt.show()

# Get number of components from user
pca_components = int(input("Enter number of components for PCA: "))
print()

# Sanity check
if 1 <= pca_components < min(M, N):
    # Re-print singular values but only include the ones in use
    print(f"~Selected Singular Value(s)~\n{S[:pca_components]}\n")

    # Project the data onto the principal components:
    matrix_projected = U @ S_expanded

    # Print the reduced dimensionality data
    print(f"~Reduced Dimensionality Data~\n{matrix_projected[:,:pca_components]}")
else:
    print("Invalid number of components!")
