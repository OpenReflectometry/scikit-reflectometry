# Example test file.

from numpy.testing import assert_equal
import skreflectometry

def test_version_good():
    assert_equal(skreflectometry.__version__, "0.1")
