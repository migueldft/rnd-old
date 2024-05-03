"""version.py tests."""

from distutils.version import StrictVersion

from src import version


class TestVersionNumber:

    def test_version(self):
        assert StrictVersion(version.__version__)
