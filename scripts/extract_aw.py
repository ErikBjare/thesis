#!/bin/env python3

from pprint import pprint
from datetime import datetime, timezone, timedelta
import json

import aw_client


def query():
    awc = aw_client.ActivityWatchClient(testing=True)
    hostname = "erb-main2-arch"

    # Basic set of categories to use as labels
    categories = [
        [["Programming"], {"type": "regex", "regex": "NVIM"}],
        [["Reading docs"], {"type": "regex", "regex": "readthedocs.io"}],
        [["StackOverflow"], {"type": "regex", "regex": "StackOverflow"}],
        [["Pull request"], {"type": "regex", "regex": "Pull Request"}],
    ]

    query = """
    events = flood(query_bucket("aw-watcher-window_{hostname}"));
    not_afk = flood(query_bucket("aw-watcher-afk_{hostname}"));
    not_afk = filter_keyvals(not_afk, "status", ["not-afk"]);
    events = filter_period_intersect(events, not_afk);
    events = categorize(events, {categories});
    RETURN = {
        "events": events
    };
    """
    query = query.replace("{hostname}", hostname)
    query = query.replace("{categories}", json.dumps(categories))
    now = datetime.now(tz=timezone.utc)
    stop = now
    start = stop - timedelta(days=2)
    print("Querying aw-server...")
    result = awc.query(query, start, stop)
    # Since we're only querying one timeperiod
    result = result[0]

    pprint(result, depth=1)
    pprint(result["events"][0])


if __name__ == "__main__":
    events = query()
