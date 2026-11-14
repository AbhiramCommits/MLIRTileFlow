#!/usr/bin/env python3
"""Check the analytical locality cost model against sanity properties.

Compares the DRAM estimate of an untiled matmul with a tiled one and verifies
that an over-capacity promoted schedule is rejected.
"""

import json
import subprocess
import sys


def main() -> int:
    tool = sys.argv[1]
    source = sys.argv[2]

    output = subprocess.check_output(
        [tool, "-tileflow-locality-report", source], text=True
    )
    report = json.loads(output.splitlines()[0])
    ops = report["ops"]
    assert ops, "no scheduled ops in report"

    by_tiles = {tuple(op["tiles"]): op for op in ops}
    untiled = by_tiles[(1024, 1024, 1024)]
    tiled = by_tiles[(32, 32, 32)]
    over = by_tiles[(512, 512, 512)]

    assert untiled["dram_bytes"] > tiled["dram_bytes"], (
        f"untiled DRAM {untiled['dram_bytes']} is not greater than tiled "
        f"DRAM {tiled['dram_bytes']}"
    )
    assert tiled["arithmetic_intensity"] > untiled["arithmetic_intensity"], (
        "tiled schedule must have higher arithmetic intensity"
    )
    assert tiled["bytes_per_level"], "missing per-level traffic"
    assert len(tiled["bytes_per_level"]) == len(report["levels"])
    assert tiled["reuse_distance_histogram"], "missing reuse histogram"

    assert not over["feasible"], "over-capacity schedule must be rejected"
    assert "scratchpad" in over.get("reason", ""), over

    print(
        "cost-model ok: untiled=%d tiled=%d over_capacity_feasible=%s"
        % (untiled["dram_bytes"], tiled["dram_bytes"], over["feasible"])
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
