import pygraphblas as gb
import numpy as np
import time
import os

# Initialize GraphBLAS
gb.init()

def create_random_matrix(rows, cols, density=0.1):
    """Create a sparse matrix with random values"""
    print(f"Creating {rows}x{cols} matrix with density {density}")
    
    # Calculate number of non-zero elements
    nnz = int(rows * cols * density)
    
    # Generate random indices
    row_indices = np.random.randint(0, rows, nnz)
    col_indices = np.random.randint(0, cols, nnz)
    values = np.random.rand(nnz).astype(np.float64)
    
    # Create PyGraphBLAS matrix
    matrix = gb.Matrix.from_coo(
        row_indices, col_indices, values, 
        nrows=rows, ncols=cols, dtype=gb.dtypes.FP64
    )
    
    return matrix

def main():
    # Matrix dimensions
    rows, cols = 1000, 1000
    
    # Create random matrix and measure time
    print("=" * 50)
    print("CREATING RANDOM MATRIX")
    print("=" * 50)
    
    start_time = time.time()
    matrix = create_random_matrix(rows, cols, density=0.1)  # 10% density
    creation_time = time.time() - start_time
    
    print(f"Matrix created successfully!")
    print(f"Shape: {matrix.shape}")
    print(f"Number of stored values: {matrix.nvals}")
    print(f"Creation time: {creation_time:.4f} seconds")
    
    # Store matrix and measure time
    print("\n" + "=" * 50)
    print("STORING MATRIX")
    print("=" * 50)
    
    filename = "matrix_1000x1000.mtx"
    
    start_time = time.time()
    # Export to Matrix Market format (standard sparse matrix format)
    matrix.to_mm(filename)
    storage_time = time.time() - start_time
    
    print(f"Matrix stored to '{filename}'")
    print(f"File size: {os.path.getsize(filename) / (1024*1024):.2f} MB")
    print(f"Storage time: {storage_time:.4f} seconds")
    
    # Load matrix back and measure time
    print("\n" + "=" * 50)
    print("LOADING MATRIX FROM STORAGE")
    print("=" * 50)
    
    start_time = time.time()
    loaded_matrix = gb.Matrix.from_mm(filename, dtype=gb.dtypes.FP64)
    loading_time = time.time() - start_time
    
    print(f"Matrix loaded from '{filename}'")
    print(f"Shape: {loaded_matrix.shape}")
    print(f"Number of stored values: {loaded_matrix.nvals}")
    print(f"Loading time: {loading_time:.4f} seconds")
    
    # Verify matrices are identical
    print("\n" + "=" * 50)
    print("VERIFICATION")
    print("=" * 50)
    
    # Check if matrices are equal (within floating point precision)
    diff = matrix - loaded_matrix
    if diff.nvals == 0:
        print("✓ Original and loaded matrices are identical!")
    else:
        print(f"⚠ Matrices differ in {diff.nvals} elements")
    
    # Summary
    print("\n" + "=" * 50)
    print("TIMING SUMMARY")
    print("=" * 50)
    print(f"Matrix creation:  {creation_time:.4f} seconds")
    print(f"Matrix storage:   {storage_time:.4f} seconds")
    print(f"Matrix loading:   {loading_time:.4f} seconds")
    print(f"Total time:       {creation_time + storage_time + loading_time:.4f} seconds")
    
    # Clean up
    try:
        os.remove(filename)
        print(f"\nCleaned up temporary file: {filename}")
    except:
        pass

if __name__ == "__main__":
    main()
