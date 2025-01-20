from typing import Dict, Iterable, List, Tuple, Union

import numpy as np

from .coordinate_utils import get_points_in_box
from .lattice import Lattice
from .constants import LOCATION, SITE_CLASS, SITE_ID, OFFSET_PRECISION

import copy

VEC_OFFSET = 0.001
DEFAULT_SITE_CLASS = "A"


class PeriodicStructure:
    """
    Represents a periodic arrangement of sites. Assigns
    identifiers to sites so that they can be referred to elsewhere.

    Supports retrieving sites by identifier or by location.

    Attributes
    ----------
    lattice : Lattice
        The periodic lattice in which this structure exists
    dim : int
        The dimensionality of the structure
    """

    @classmethod
    def build_from(
        _,
        lattice: Lattice,
        num_cells: List[int],
        site_motif: Union[Dict, List],
        frac_coords: bool = False,
    ):
        """Builds a PeriodicStructure by repeating the unit cell num_cell times
        in each dimension. For instance, to build a structure that has 2 unit cells in
        each direction (and itself is three dimensional), the num_cells parameter should be

        [2, 2, 2]

        Lattice sites must also be specified by the site_motif parameter. This allows
        specification of which sites are where in the unit cell. For instance, if my 2D
        lattice has two types of sites, A and B, and each type exists in two different
        places, I might use the following site_motif:

        {
            "A": [
                [0.2, 0.2],
                [0.4, 0.4]
            ],
            "B": [
                [0.6, 0.6],
                [0.8, 0.8]
            ]
        }

        Parameters
        ----------
        num_cells : List[int]
            As described above, a list of the number of repetitions of the unit cell
            in each direction.
        site_motif : dict
            A dictionary mapping string site classes to lists of the locations within
            the unit cell at which a site of that type exists.

        Returns
        -------
        PeriodicStructure
            The structure resulting from the lattice tiling and motif filling specified.
        """
        new_lattice = lattice.get_scaled_lattice(num_cells)

        struct = PeriodicStructure(new_lattice)

        if not isinstance(site_motif, dict):
            site_motif = {DEFAULT_SITE_CLASS: site_motif}

        # these are in "fractional" coordinates
        vec_coeffs = get_points_in_box([0 for _ in range(new_lattice.dim)], num_cells)
        for vec_coeff_set in vec_coeffs:
            if not frac_coords:  # convert lattice points to "cartesian coordinates"
                point = lattice.matrix @ np.array(vec_coeff_set)
            else:
                point = np.array(vec_coeff_set)

            for site_class, basis_vecs in site_motif.items():
                for vec in basis_vecs:
                    # if the motif is specified in cartesian coordinates, we're good here
                    site_loc = tuple(point + np.array(vec))

                    if (
                        frac_coords
                    ):  # convert lattice point back to cartesian coordinates
                        site_loc = lattice.get_cartesian_coords(site_loc)

                    # site_loc should be in cartesian coordinates at this point
                    struct.add_site(site_class, site_loc)

        return struct

    def __init__(self, lattice: Lattice):
        """Instantiates a structure with the specified lattice.
        The dimensionaliity is inferred by the dimensionality of the lattice.

        Parameters
        ----------
        lattice : Lattice
            _description_
        """
        self.lattice = lattice
        self.dim = lattice.dim
        self._sites = {}
        self._site_ids = []
        self._location_lookup = {}
        self._offset_vector = np.array([VEC_OFFSET for _ in range(self.dim)])

    def as_dict(self):
        copied = copy.deepcopy(self._sites)
        for _, site in copied.items():
            site[LOCATION] = site[LOCATION].tolist()

        return {
            "lattice": self.lattice.as_dict(),
            "_sites": copied,
        }

    @property
    def site_ids(self):
        return copy.copy(self._site_ids)

    @classmethod
    def from_dict(cls, d):
        struct = cls(Lattice.from_dict(d["lattice"]))
        sites = {int(k): v for k, v in d["_sites"].items()}

        for _, site in sites.items():
            struct.add_site(site[SITE_CLASS], site[LOCATION])

        return struct

    def _get_rounded_coords(self, location: Iterable[float]) -> Iterable[float]:
        return np.round(location, OFFSET_PRECISION)

    def _coords_with_offset(self, location: Iterable[float]) -> Iterable[float]:
        return self._get_rounded_coords(location + self._offset_vector)

    def _transformed_coords(self, location: Iterable[float]) -> Iterable[float]:
        periodized_coords = self.lattice.get_periodized_cartesian_coords(location)
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
            The location of the new site in Cartesian coordinates

        Returns
        -------
        int
            The ID of the site. This can be used to retrieve the site later
        """
        new_site_id = len(self._sites)

        periodized_coords = self._get_rounded_coords(
            self.lattice.get_periodized_cartesian_coords(location)
        )
        offset_periodized_coords = tuple(self._transformed_coords(location))

        assert (
            self._location_lookup.get(offset_periodized_coords, None) is None
        ), "That site is already occupied"

        self._sites[new_site_id] = {
            SITE_CLASS: site_class,
            LOCATION: periodized_coords,
            SITE_ID: new_site_id,
        }

        self._location_lookup[offset_periodized_coords] = new_site_id
        self._site_ids.append(new_site_id)
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
        _transformed_coords = tuple(self._transformed_coords(location))
        site_id = self._location_lookup.get(_transformed_coords)

        if site_id is not None:
            return self.get_site(site_id)
        else:
            return None

    def id_at(self, location: Tuple[float]) -> Dict:
        site = self.site_at(location)
        if site is None:
            return None
        else:
            return site[SITE_ID]

    def class_at(self, location: Tuple[float]) -> Dict:
        site = self.site_at(location)
        if site is None:
            return None
        else:
            return site[SITE_CLASS]

    def site_class(self, site_id: int) -> str:
        return self.get_site(site_id)[SITE_CLASS]

    def site_location(self, site_id: int) -> str:
        return self.get_site(site_id)[LOCATION]

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

    def all_site_classes(self) -> List[str]:
        """Returns a list of all the site classes present in this structure.

        Returns
        -------
        List[str]
            The site classes in this structure. Each class appears once in this list.
        """
        return list({site[SITE_CLASS] for site in self.sites()})

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

        return [site for site in all_sites if site[SITE_CLASS] == site_class]
