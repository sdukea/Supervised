# learn all about NumPy, its application on vectorizaton and all things related

import numpy as np

# vectors
# - ordered array of numbers
# - lower case bold letters
# - elements in vector are all the same type; does not contain characters and numbers ATST
#   a 2D vector has a dimension of 2 i.e 2 values in the vector (not a matrix with rows AND columns)
#   a 3D vector has a dimension of 3 i.e. 3 values in the vector
#   a nD vector has n dimensions i.e n values inside the vector
# - indexing a vector can be done; from 0 to n-1 in code; 1 to n normally
# - also, this is what RANK means NOTE: In NumPy:
#   rank 1: np.array([1, 2, 3])
#   rank 2: np.array([[1,2], [3,4]])
#   rank 3: np.array([[[1],[2]], [[3],[4]]])
#   its the number of '[]' you have when creating an array - the rank
# - dimension also means: the number of features/rows (.shape[0] as well; yields no: of rows/feat.)

# numpy arrays
# - numpy's basic data structure
# - is an indexable, n-dimensional array
# - containing elements of the same datatype
# - NOTE: In numpy, DIMENSION refers to the number of indexes of an array
#   a 1-D array has one index
#   - x = np.array([1.0, 2.0, 3.0])
#   - x's shape -> (3,)
#   - and you index 1-D arrays only with one set of '[]'
#   - and hence, dimension is only one because there is only one index (one set of '[]'/index)
#   for a 2-D array like
#   - x = np.array([[1.0, 2.0], [3.0, 4.0]])
#   - the shape is -> (2,2)
#   - because x is actually
#   - [1.0, 2.0]
#   - [3.0, 4.0]
#   - and you index with 2 index '[]'
#   - x[1][0] - indexing the first element out of the second element in the 2-D array
#   - and the same for 3-D array -> 3 '[]' and in NumPy, the number of indexes is dimension
#   - NOTE: In math, DIMENSIONS are the independent directions you can take
#   - So, [1.0, 2.0, 3.0] lives in a 3-D space
#   - There is no 'dimension' of a matrix - you say its rows x columns
#   - For a matrix, you can maybe tell the dimension of the rows because rows itself is a vector
#   - i.e. a row vector (and maybe tell the dimension of its column/column vectors as well)
#   - NOTE: In math, RANK is the number of linearly independent rows/columns
#   - For:
#   - [1.0  2.0]
#   - [3.0  4.0]
#   - the rank is 2 because there are two independent rows or columns IN A MATRIX
#   - The above definition for NumPy actually mirrors the idea that rank applies for a matrix because
#   - having multiple '[]' actually lets you build a matrix (think and imagine)
#   - If one row can be made from others, its redundant - does not contribute to rank.
#   - rank is basically how many unique directions your data actually spans
#   - [1.0  2.0]
#   - [3.0  4.0]
#   - Neither of the rows depend on each other - so they're independen
#   - and the rank of this matrix is 2
#   - But when it comes to this:
#   - [1.0  2.0]
#   - [2.0  4.0]
#   - Row 2 is 2 x row 1
#   - and row 2 is dependent on row 1 - rank is only 1 here as row 2 is redundant

