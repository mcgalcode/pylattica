import pytest

from pylattica.core.runner import merge_updates
from pylattica.core.constants import GENERAL, SITES

@pytest.fixture
def curr_updates():
    return { SITES: { 0: { "a": 1 }}, GENERAL: {} }

def test_merge_updates_no_new(curr_updates):
    new_updates =  None
    updated_updates = merge_updates(new_updates, curr_updates)
    assert updated_updates == curr_updates

def test_merge_updates_full_with_sites(curr_updates):

    new_updates = { SITES: { 1: { "b": 1 }} }
    updated_updates = merge_updates(new_updates, curr_updates)

    expected = {
        SITES: {
            1: {"b": 1},
            0: {"a": 1}
        },
        GENERAL: {}
    }

    assert updated_updates == expected

def test_merge_updates_full_with_sites_and_general(curr_updates):

    new_updates = { SITES: { 1: { "b": 1 }}, GENERAL: { "c": 2 } }
    updated_updates = merge_updates(new_updates, curr_updates)

    expected = {
        SITES: {
            1: {"b": 1},
            0: {"a": 1}
        },
        GENERAL: {
            "c": 2
        }
    }

    assert updated_updates == expected

def test_merge_updates_full_with_sites_overwrite(curr_updates):

    new_updates = { SITES: { 1: { "b": 1 }, 0: { "a": 2 }} }
    updated_updates = merge_updates(new_updates, curr_updates)

    expected = {
        SITES: {
            1: {"b": 1},
            0: {"a": 2}
        },
        GENERAL: {}
    }

    assert updated_updates == expected

def test_merge_updates_implicit_sites(curr_updates):

    new_updates = { 1: { "b": 1 }}
    updated_updates = merge_updates(new_updates, curr_updates)

    expected = {
        SITES: {
            1: {"b": 1},
            0: {"a": 1}
        },
        GENERAL: {}
    }

    assert updated_updates == expected

def test_merge_updates_implicit_sites_overwrite(curr_updates):

    new_updates = { 1: { "b": 1 }, 0: { "a": 2 }}
    updated_updates = merge_updates(new_updates, curr_updates)

    expected = {
        SITES: {
            1: {"b": 1},
            0: {"a": 2}
        },
        GENERAL: {}
    }

    assert updated_updates == expected

def test_merge_updates_specific_site(curr_updates):

    new_updates = { "b": 1 }
    updated_updates = merge_updates(new_updates, curr_updates, site_id=1)

    expected = {
        SITES: {
            1: {"b": 1},
            0: {"a": 1}
        },
        GENERAL: {}
    }

    assert updated_updates == expected

def test_merge_updates_specific_site_overwrite(curr_updates):

    new_updates = { "a": 2 }
    updated_updates = merge_updates(new_updates, curr_updates, site_id=0)

    expected = {
        SITES: {
            0: {"a": 2}
        },
        GENERAL: {}
    }

    assert updated_updates == expected