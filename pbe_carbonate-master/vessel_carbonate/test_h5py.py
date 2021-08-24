import pytest
import numpy as np
import h5py_utils
import h5py

# HDF5_FILE = h5py.File('aa.h5', 'w')

@pytest.fixture()
def dic_plain():
    return {
        'aow': np.array([[1,2], [0.432, 0.21]]),
        'c': 2.0,
        'b': 5.0
    }

def test_dic_plain(dic_plain):
    HDF5_FILE = h5py.File('aa.h5', 'w')
    d = dic_plain
    h5py_utils.save_dict_to_file(HDF5_FILE, d)
    HDF5_FILE.close()

@pytest.fixture()
def dic_mix1():
    return {
        'aow': np.array([[1,2], [0.432, 0.21]]),
        'c': [1, 4.0, 12.0, -23924.032],
        'b': 5.0
    }

def test_dic_mix1(dic_mix1):
    HDF5_FILE = h5py.File('aa.h5', 'w')
    d = dic_mix1
    h5py_utils.save_dict_to_file(HDF5_FILE, d)
    HDF5_FILE.close()

@pytest.fixture()
def dic_mix2():
    return {
        'array2d': np.array([[1,2], [0.432, 0.21]]),
        'array1d': np.linspace(0, 200, num=200, dtype=int),
        'scalar': 5.0,
        'list-scalar': np.linspace(0, 200, num=200, dtype=int).tolist(),
        'list-dict': [
            {'dict': {'a': np.array([[1,2], [12.2, 1e-3]])}},
            {'list': [1, 'fdfod', 'alow']}
        ],
    }

def test_dic_mix2(dic_mix2):
    HDF5_FILE = h5py.File('aa.h5', 'w')
    d = dic_mix2
    h5py_utils.save_dict_to_file(HDF5_FILE, d)
    HDF5_FILE.close()



def test_key_with_zeros():
    new = h5py_utils.key_with_zeros(0, 10)
    assert new == '0'
    new = h5py_utils.key_with_zeros(9, 10)
    assert new == '9'
    new = h5py_utils.key_with_zeros(10, 11)
    assert new == '10'
    new = h5py_utils.key_with_zeros(1, 11)
    assert new == '01'
    new = h5py_utils.key_with_zeros(2, 100)
    assert new == '02'
    new = h5py_utils.key_with_zeros(2, 101)
    assert new == '002'
    new = h5py_utils.key_with_zeros(59, 101)
    assert new == '059'

if __name__ == "__main__":
    test_dic_mix1(dic_mix1())
