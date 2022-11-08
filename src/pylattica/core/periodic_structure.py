from functools import cache
from numbers import Number
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .constants import LOCATION, SITE_CLASS, SITE_ID


@cache
def get_pt_in_range(bound: float, pt: float) -> float:
    """Returns the periodic image of a 1D value.

    Parameters
    ----------
    bound : float
        The upper bound of the range (lower bound is assumed zero)
    pt : float
        The value to return to within the range

    Returns
    -------
    float
        The transformed value
    """
    return pt % bound


@cache
def get_periodic_point(bounds: Iterable[Number], pt: Iterable[Number]) -> tuple[Number]:
    """Given a point and a list of upper bounds (assuming zero lower bounds),
    returns the point, unchanged, if it lies inside the two or three
    dimensional square region defined by the bounds provided, or the periodic
    image of the point within the bounds if the point lies outside the bounds.

    For example, if the bounds are (5, 5, 5) and the point is (2, 3, 6),
    the periodized version of the point is (2, 3, 1).

    Parameters
    ----------
    bounds : Iterable[Number]
        The upper boundaries along each dimension of the periodic image
    pt : Iterable[Number]
        The point to move into the origina image

    Returns
    -------
    tuple[Number]
        The transformed point
    """
    return tuple([get_pt_in_range(b, p) for b, p in zip(bounds, pt)])


def float_loc(loc: Iterable[Number]) -> Tuple[float]:
    """Returns a new list with each element of an iterable of numerics
    cast as a float.

    Parameters
    ----------
    loc : Iterable[Number]
        A location represented by an arbitrary iterable of numbers

    Returns
    -------
    Tuple[float]
        The same location represented as a tuple of floats
    """
    return tuple(loc)


OFFSET_PRECISION = 3
VEC_OFFSET = 0.001


class PeriodicStructure:
    """
    Represents a periodic arrangement of sites. Assigns
    identifiers to sites so that they can be referred to elsewhere.

    Supports retrieving sites by identifier or by location.

    Attributes
    ----------
    bounds : Iterable[Number]
        The extent of this structure in each dimension in real units. Lower bounds are
        assumed to be zero. For instance, a cubic structure of length 2 in
        each direction would have bounds of (2, 2, 2)
    dim : int
        The dimensionality of the structure
    """

    def __init__(self, bounds: Iterable[Number]):
        """Instantiates a structure with the specified bounds.
        The dimensionaliity is inferred by the dimensionality of the bounds.

        Parameters
        ----------
        bounds : Iterable[Number]
            _description_
        """
        self.bounds = tuple(bounds)
        self.dim = len(bounds)
        self._sites = {}
        self.site_ids = []
        self._location_lookup = {}
        self._offset_vector = np.array([VEC_OFFSET for _ in range(self.dim)])

    def periodized_coords(self, location: Tuple[float]) -> Tuple[float]:
        """Returns the periodic image of a point within this structure.

        Parameters
        ----------
        location : Tuple[float]
            The point to be periodi

        Returns
        -------
        Tuple[float]
            The periodized point
        """
        periodic_point = get_periodic_point(self.bounds, location)
        return periodic_point

    def _coords_with_offset(self, location: Iterable[float]) -> Iterable[float]:
        return tuple([s + VEC_OFFSET for s in location])

    def _transformed_coords(self, location: Iterable[float]) -> Iterable[float]:
        periodized_coords = self.periodized_coords(location)
        offset_periodized_coords = self._coords_with_offset(periodized_coords)
        return offset_periodized_coords

    def add_site(self, site_class: str, location: Tuple[float]) -> int:
        """Adds a new site to the structure.

        Parameters
        ----------
        site_class : str
            The class of the site to be added. This can be anything, but must be provided
            Think of this as a tag for the site
        location : Tuple[float]
            The location of the new site

        Returns
        -------
        int
            The ID of the site. This can be used to retrieve the site later
        """
        new_site_id = len(self._sites)
        periodized_coords = self.periodized_coords(location)

        offset_periodized_coords = self._coords_with_offset(periodized_coords)

        assert (
            self._location_lookup.get(offset_periodized_coords, None) is None
        ), "That site is already occupied"

        self._sites[new_site_id] = {
            SITE_CLASS: site_class,
            LOCATION: periodized_coords,
            SITE_ID: new_site_id,
        }

        self._location_lookup[offset_periodized_coords] = new_site_id
        self.site_ids.append(new_site_id)
        return new_site_id

    def site_at(self, location: Tuple[float]) -> Dict:
        """Retrieves the site at a particular location. Uses float equality to check.

        Parameters
        ----------
        location : Tuple[float]
            The location of the desired site

        Returns
        -------
        int
            A dictionary with keys "site_class", "location", and "id" representing the site.
        """
        location = tuple(location)
        _transformed_coords = self._transformed_coords(location)
        site_id = self._location_lookup.get(_transformed_coords)
        if site_id is not None:
            return self.get_site(site_id)
        else:
            return None

    def get_site(self, site_id: int) -> Dict:
        """Returns the site with the specified ID.

        Parameters
        ----------
        site_id : int
            The ID of the desired site.

        Returns
        -------
        Dict
            A dictionary with keys "site_class", "location", and "id" representing the site.
        """
        return self._sites.get(site_id)

    def sites(self, site_class: str = None) -> List[Dict]:
        """Returns a list of the sites with the specified site class.

        Parameters
        ----------
        site_class : str, optional
            The desired site class, by default None

        Returns
        -------
        List[Dict]
            A list of the sites matching that site class.
        """
        all_sites = list(self._sites.values())
        if site_class is None:
            return all_sites
        else:
            return [site for site in all_sites if site[SITE_ID] == site_class]
