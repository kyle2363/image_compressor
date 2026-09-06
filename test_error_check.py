import compressor
import numpy as np
from PIL import Image

def test_simple_array_error():
    test_arr1 = np.array([[[1,1,1]]])
    test_arr2 = np.array([[[2,2,2]]])
    error = compressor.error_percent_simple(test_arr1, test_arr2, 0,0,1,1)
    assert error == 1

def test_2d_array_error():
    test_arr1 = np.array([[[1,1,1], [1,1,1]], [[1,1,1], [1,1,1]]])
    test_arr2 = np.array([[[2,2,2], [2,2,2]], [[2,2,2], [2,2,2]]])
    error = compressor.error_percent_simple(test_arr1, test_arr2, 0,0,2,2)
    assert error == 1

def test_2d_array_error():
    test_arr1 = np.array([[[1,1,1], [1,1,1]], [[1,1,1], [1,1,1]]])
    test_arr2 = np.array([[[2,1,1], [1,1,1]], [[1,1,1], [1,1,1]]])
    error = compressor.error_percent_simple(test_arr1, test_arr2, 0,0,2,2)
    assert np.isclose(error, 0.08333, 0.001)

def test_compress_0_simple():
    test_arr1 = np.array([[[1,1,1], [1,1,1]], [[1,1,1], [1,1,1]]])
    result_arr = compressor.process_image(np.copy(test_arr1), 0.001, 0)
    np.testing.assert_array_equal(result_arr, test_arr1)

def test_compress_0_simple():
    test_arr1 = np.array([[[1,2,3], [1,2,3]], [[1,2,3], [1,2,3]]])
    result_arr = compressor.process_image(np.copy(test_arr1), 0.00001, 0)
    np.testing.assert_array_equal(result_arr, test_arr1)

def test_compress_0_simple_blue():
    img = Image.open("Solid_blue.png")
    img.resize((25,25))
    img_arr = np.array(img)
    result_arr = compressor.process_image(np.copy(img_arr), 0.0001, 0)
    np.testing.assert_array_equal(result_arr, desired= img_arr)

def test_compress_0_simple_darkCity():
    size = 8
    img = Image.open("dark_city.png").resize((size,size))
    img_arr = np.array(img)
    result_arr = compressor.process_image(np.copy(img_arr), 0.1, 0)
    assert compressor.error_percent_simple(img_arr, result_arr,0,0,size, size) < 0.001