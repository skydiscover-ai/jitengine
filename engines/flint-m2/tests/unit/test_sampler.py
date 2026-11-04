"""The seeded per-(seed, position) uniform that sampled verify relies on."""

from jitengine_flint_m2.sampler import uniform


def test_sampler_uniform():
    a = [uniform(7, j) for j in range(1000)]
    assert a == [uniform(7, j) for j in range(1000)]
    assert all(0.0 <= x < 1.0 for x in a)
    assert a != [uniform(8, j) for j in range(1000)]
    assert 0.45 < sum(a) / len(a) < 0.55
