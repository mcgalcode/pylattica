import math
import itertools

def get_in_range(upper, val):
    if val < 0:
        dist = -val - 1
        rem = math.remainder(dist, upper)
        return int(upper - rem)
    elif val > upper:
        dist = val - upper - 1
        rem = math.remainder(dist, upper)
        return int(rem)
    else:
        return int(val)

def get_coords_in_box(size, coords):
    highest_val = size - 1
    return tuple([get_in_range(highest_val, c) for c in coords])

def get_points_in_box(lb, ub, dim):
    vals = list(range(lb, ub))
    args = [vals for _ in range(dim)]
    return list(itertools.product(*args))
