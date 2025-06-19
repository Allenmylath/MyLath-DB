import graphblas as gb
# Test the actual operation that's failing:
try:
    test_matrix = gb.Matrix(gb.dtypes.BOOL, nrows=2, ncols=2)
    test_matrix[0, 1] = True
    
    test_vector = gb.Vector(gb.dtypes.BOOL, size=2)
    test_vector[0] = True
    
    # Try different semirings:
    for semiring_name in ['any_pair', 'lor_land', 'max_min']:
        try:
            semiring = getattr(gb.semiring, semiring_name)
            result = test_vector.vxm(test_matrix, semiring)
            print(f"✅ SUCCESS with gb.semiring.{semiring_name}")
            print(f"   Result nvals: {result.nvals}")
            break
        except Exception as e:
            print(f"❌ FAILED with gb.semiring.{semiring_name}: {e}")
            
except Exception as e:
    print(f"❌ Test setup failed: {e}")