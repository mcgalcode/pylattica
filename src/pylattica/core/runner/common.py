from ..constants import GENERAL, SITES


def merge_updates(new_updates, curr_updates=None, site_id=None):
    if new_updates is None:
        return curr_updates

    if curr_updates is None:
        curr_updates = {SITES: {}, GENERAL: {}}

    # if this is a total
    if SITES in new_updates or GENERAL in new_updates:
        curr_updates[SITES].update(new_updates.get(SITES, {}))
        curr_updates[GENERAL].update(new_updates.get(GENERAL, {}))
    # if these updates only include site updates
    elif set(map(type, new_updates.keys())) == {int}:
        curr_updates[SITES].update(new_updates)
    # if these updates only apply to a single site
    elif site_id is not None:
        curr_updates[SITES].update({site_id: new_updates})
    else:
        raise ValueError("Bad combination of arguments for merge_updates")

    return curr_updates
