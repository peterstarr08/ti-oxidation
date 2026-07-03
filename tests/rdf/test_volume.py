from ti_oxidation.rdf.volume import *
from math import pi

import pytest

# Get sphere volume
@pytest.mark.parametrize(
            'r, volume',
                [
                    (0, 0),
                    (3, 4/3*pi*3**3),
                    pytest.param(4, 121, marks=pytest.mark.xfail)
                ]
            )
def test_get_sphere_volume(r, volume):
    assert get_sphere_volume(r)==volume



# Get cap volume
@pytest.mark.parametrize(
            'r, h, volume',
            [
                (2, 0, 0),
                (2, 2*2, get_sphere_volume(2)),
                (4, 4, get_sphere_volume(4)/2)
                ]
        )
def test_get_cap_volume(r, h, volume):
    assert get_cap_volume(r, h)==volume


# Case 1: r>h
# Case 2: r=h
# Case 3: r<h
# Get effective volume
@pytest.mark.parametrize(
            'r, h, z, volume',
            [
                (2, 1, 0, get_sphere_volume(2)/2 - get_cap_volume(2, 1)),
                (2, 1, 0.5, (get_sphere_volume(2) - get_cap_volume(2, 1.5)*2)),
                (2, 1, 1, get_sphere_volume(2)/2 - get_cap_volume(2, 1)),

                (1, 1, 0, get_sphere_volume(1)/2),
                (1, 1, 0.5, get_sphere_volume(1) - 2*5/24*pi*1**3),
                (1, 1, 1, get_sphere_volume(1)/2),

                (1, 2, 0, get_sphere_volume(1)/2),
                (1, 2, 1, get_sphere_volume(1)),
                (1, 2, 2, get_sphere_volume(1)/2),

                # Some random shit
                (2, 3, 1, get_sphere_volume(2) - get_cap_volume(2, 1) - 0),
                (2, 3, 2, get_sphere_volume(2) - get_cap_volume(2, 1) - 0)
            ]
        )
def test_get_effective_volume(r, h, z, volume):
    assert get_effective_volume(r, h, z) == pytest.approx(volume, rel=0, abs=1e-14)
