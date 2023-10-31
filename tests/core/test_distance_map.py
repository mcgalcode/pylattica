from pylattica.core.distance_map import EuclideanDistanceMap, ManhattanDistanceMap

def test_distance_map_basics():
    dmap = EuclideanDistanceMap([
        (1, 1),
        (0, 1)
    ])

    assert dmap.get_dist((1, 1)) == 1.41
    assert dmap.get_dist((0, 1)) == 1


def test_manhattan_distance_map():
    dmap = ManhattanDistanceMap([
        (1,1),
        (0,1),
        (0,-1),
        (2.5,1.7)
    ])

    assert dmap.get_dist((1,1)) == 2
    assert dmap.get_dist((0,1)) == 1
    assert dmap.get_dist((0,-1)) == 1
    assert dmap.get_dist((2.5,1.7)) == 4.2