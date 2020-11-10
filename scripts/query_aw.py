#!/bin/env python3

"""
Script to extract activity-labels from ActivityWatch to be used when training classifier.
"""

from copy import deepcopy
from typing import List, Callable
from pprint import pprint
from datetime import datetime, timezone, timedelta
import json

import aw_client
from aw_core import Event
from timeslot import Timeslot

UNCAT = ["Uncategorized"]


def query() -> List[Event]:
    awc = aw_client.ActivityWatchClient(testing=True)
    hostname = "erb-main2-arch"

    now = datetime.now(tz=timezone.utc)
    stop = now
    start = stop - timedelta(days=2)

    def cat_re(re_str):
        return {"type": "regex", "regex": re_str}

    # Basic set of categories to use as labels
    categories = [
        [["Programming"], cat_re("NVIM")],
        [["Reading docs"], cat_re("readthedocs.io")],
        [["Stack Overflow"], cat_re("Stack Overflow")],
        [["Pull request"], cat_re("Pull Request")],
    ]

    query = """
    events = flood(query_bucket("aw-watcher-window_{hostname}"));
    not_afk = flood(query_bucket("aw-watcher-afk_{hostname}"));
    not_afk = filter_keyvals(not_afk, "status", ["not-afk"]);
    events = filter_period_intersect(events, not_afk);
    events = categorize(events, {categories});
    cat_events = sort_by_duration(merge_events_by_keys(events, ["$category"]));
    RETURN = {
        "events": events,
        "duration_by_cat": cat_events
    };
    """

    # Insert parameters
    # (not done with f-strings since they don't like when there's other {...}'s in the string, and I don't want to {{...}})
    query = query.replace("{hostname}", hostname)
    query = query.replace("{categories}", json.dumps(categories))

    print("Querying aw-server...")
    result = awc.query(query, start, stop)

    # Since we're only querying one timeperiod
    result = result[0]

    # pprint(result, depth=1)
    # pprint(result["events"][0])

    # Transform to Event
    events = [Event(**e) for e in result["events"]]
    duration_by_cat = [Event(**e) for e in result["duration_by_cat"]]

    print("Time by category:")
    print_events(duration_by_cat, lambda e: e["data"]["$category"])

    return events


def print_events(events, label: Callable[[Event], str]):
    for e in events:
        print(f"{e.duration.total_seconds() : >8.1f}s - {e['data']['$category']}")


def merge_adjacent_by_category(events) -> List[Event]:
    """Merge adjacent events with same category"""
    # To ensure no side-effects
    events = deepcopy(events)

    # TODO: Is this gap size reasonable?
    gap_allow = timedelta(seconds=10)  # Gaps up to 10s allowed

    cat_events = [events[0]]
    for e in events[1:]:
        prev_event = cat_events[-1]
        if e.data["$category"] == prev_event.data["$category"]:
            prev_end = prev_event.timestamp + e.duration
            if prev_end + gap_allow > e.timestamp:
                # Extend prev event to new end
                prev_event.duration = (e.timestamp + e.duration) - prev_event.timestamp
                # print("Merged event")
            else:
                # Event was not within gap to be considered adjacent
                cat_events.append(e)
        elif e.data["$category"] == UNCAT:
            # Event was uncategorized, skip
            # NOTE: Gap allows for uncategorized time to be categorized by being close to another category
            continue
        else:
            cat_events.append(e)

    # Clean events, since app/title are corrupted when merged anyway
    for e in cat_events:
        del e.data["title"]
        del e.data["app"]

    return cat_events


def main() -> None:
    events = query()
    catset = {tuple(e["data"]["$category"]) for e in events}
    print(f"Categories: {catset}")

    cat_events = merge_adjacent_by_category(events)

    # Print category series
    # print_events(cat_events, lambda e: e["data"]["$category"])

    # TODO: If total uncategorized duration over a certain %, show largest uncategorized events
    # pprint(cat_events)

    # Print largest events
    # These events mark suitable timeperiods to use for the training set
    print("\nLargest categorized events:")
    largest_events = sorted(
        [e for e in cat_events if e.data["$category"] != UNCAT],
        key=lambda e: -e.duration,
    )
    entries = [
        (Timeslot(e.timestamp, e.timestamp + e.duration), e.data["$category"])
        for e in largest_events
    ]
    pprint(
        [
            (t[0].start.isoformat(), t[0].duration.total_seconds(), t[1])
            for t in entries[:5]
        ]
    )


if __name__ == "__main__":
    main()
