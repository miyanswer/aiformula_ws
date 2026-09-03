from ament_pep257.main import main
import pytest


@pytest.mark.pep257
@pytest.mark.linter
def test_pep257():
    rc = main(argv=[])
    assert rc == 0, 'Found code style errors / warnings'
