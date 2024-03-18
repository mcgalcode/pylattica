import pytest

from pylattica.visualization import DiscreteCellArtist
from pylattica.core import SimulationState


def test_discrete_cell_artist_no_legend_no_cmap():
    phases = ["a", "b"]
    artist = DiscreteCellArtist.from_phase_list(phases, state_key="x")

    assert artist.get_color_from_cell_state({"x": "a"}) != (0, 0, 0)
    assert artist.get_color_from_cell_state({"x": "c"}) == (0, 0, 0)
    assert artist.get_color_from_cell_state({"x": "b"}) != (0, 0, 0)
    assert artist.get_color_from_cell_state(
        {"x": "b"}
    ) != artist.get_color_from_cell_state({"x": "a"})

    assert artist.get_cell_legend_label({"x": "a"}) == "a"
    assert artist.get_cell_legend_label({"x": "b"}) == "b"
    assert artist.get_cell_legend_label({"x": "c"}) == "c"

    state = SimulationState(
        {
            "SITES": {
                1: {"x": "a"},
                2: {"x": "b"},
            }
        }
    )

    legend = artist.get_legend(state)

    assert "a" in legend
    assert "b" in legend

    assert legend.get("a") == artist.get_color_from_cell_state({"x": "a"})
    assert legend.get("b") == artist.get_color_from_cell_state({"x": "b"})


def test_discrete_cell_artist_no_legend_cmap():
    a_color = (50, 60, 70)
    b_color = (110, 120, 130)

    cmap = {"a": a_color, "b": b_color}

    artist = DiscreteCellArtist(cmap, state_key="x")
    assert artist.get_color_from_cell_state({"x": "a"}) == a_color
    assert artist.get_color_from_cell_state({"x": "c"}) == (0, 0, 0)
    assert artist.get_color_from_cell_state({"x": "b"}) == b_color

    assert artist.get_cell_legend_label({"x": "a"}) == "a"
    assert artist.get_cell_legend_label({"x": "b"}) == "b"
    assert artist.get_cell_legend_label({"x": "c"}) == "c"

    state = SimulationState(
        {
            "SITES": {
                1: {"x": "a"},
                2: {"x": "b"},
            }
        }
    )

    legend = artist.get_legend(state)

    assert "a" in legend
    assert "b" in legend

    assert legend.get("a") == a_color
    assert legend.get("b") == b_color


def test_discrete_cell_artist_legend_and_cmap():
    a_color = (50, 60, 70)
    b_color = (110, 120, 130)

    a_color_leg = (200, 210, 220)
    b_color_leg = (230, 240, 250)

    cmap = {"a": a_color, "b": b_color}

    legend = {"a": a_color_leg, "b": b_color_leg}

    artist = DiscreteCellArtist(cmap, state_key="x", legend=legend)
    assert artist.get_color_from_cell_state({"x": "a"}) == a_color
    assert artist.get_color_from_cell_state({"x": "c"}) == (0, 0, 0)
    assert artist.get_color_from_cell_state({"x": "b"}) == b_color

    assert artist.get_cell_legend_label({"x": "a"}) == "a"
    assert artist.get_cell_legend_label({"x": "b"}) == "b"
    assert artist.get_cell_legend_label({"x": "c"}) == "c"

    state = SimulationState(
        {
            "SITES": {
                1: {"x": "a"},
                2: {"x": "b"},
            }
        }
    )

    legend = artist.get_legend(state)

    assert "a" in legend
    assert "b" in legend

    assert legend.get("a") == a_color_leg
    assert legend.get("b") == b_color_leg
