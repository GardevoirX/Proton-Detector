import os
import sys

import numpy as np
import pytest

test_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(test_dir, '..')))

def pytest_configure():
    pytest.TOPFILE = os.path.join(test_dir, "../data/top.xyz")
    pytest.TRAJFILE = os.path.join(test_dir, "../data/traj.xtc")
    pytest.CELL = np.array([19.728, 19.728, 19.728, 90, 90, 90])
    pytest.ref_pos0 = np.array([17.51000023, 14.87000084, 15.07999992])
    pytest.ref_pos1 = np.array([13.03999996, 14.06000042, 17.22000122])