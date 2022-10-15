from pylattica.core.distance_map import EuclideanDistanceMap

def test_distance_map_basics():
    dmap = EuclideanDistanceMap([
        (1, 1),
        (0, 1)
    ])

    assert dmap.get_dist((1, 1)) == 1.41
    assert dmap.get_dist((0, 1)) == 1