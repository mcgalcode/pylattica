from numbers import Number
from typing import Dict, Iterable, List, Tuple

from .constants import LOCATION, SITE_CLASS, SITE_ID

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
    return tuple([round(float(coord), 3) for coord in loc])

class PeriodicStructure():
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
    unit_cell_size : Iterable[int]
        The extent of this structure in each dimension in unit cell counts. For instance,
        a structure that stretches for 3 unit cells in the x and y directions, and 2 unit
        cells in the z direction would have unit_cell_size of (3, 3, 2)
    """

    def __init__(self, bounds: Iterable[Number], unit_cell_size: Iterable[Number]):
        """Instantiates a structure with the specified bounds and unit_cell_size.
        The dimensionaliity is inferred by the dimensionality of the two parameters.

        Parameters
        ----------
        bounds : Iterable[Number]
            _description_
        unit_cell_size : Iterable[Number]
            _description_
        """        
        self.bounds = bounds
        self.unit_cell_size = unit_cell_size
        self.dim = len(bounds)
        self._sites = {}
        self._location_lookup = {}


    def periodized_coords(self, location: Tuple[float]) -> Tuple[float]:
        """Returns the periodic image of a point within this 

        Parameters
        ----------
        location : Tuple[float]
            The point to be periodi

        Returns
        -------
        Tuple[float]
            The periodized point
        """        
        float_coords = float_loc(location)
        return float_loc(get_periodic_point(self.bounds, float_coords))

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

        assert self._location_lookup.get(periodized_coords, None) is None, "That site is already occupied"

        self._sites[new_site_id] = {
            SITE_CLASS: site_class,
            LOCATION: periodized_coords,
            SITE_ID: new_site_id,
        }

        self._location_lookup[periodized_coords] = new_site_id
        return new_site_id

    def site_at(self, location: Tuple[float]) -> int:
        """Retrieves the site at a particular location. Uses float equality to check.

        Parameters
        ----------
        location : Tuple[float]
            The location of the desired site

        Returns
        -------
        int
            The ID of the new site.
        """        
        periodized_coords = self.periodized_coords(location)
        site_id = self._location_lookup.get(periodized_coords)
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
