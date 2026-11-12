#!/usr/bin/env python3
"""Convert an old (v1, 768-input) Endgame network file to the current v2 format.

The result evaluates every position exactly like the old network (every king bucket and every
output head starts as a copy), so the engine's strength is unchanged until the network is
trained further with `trainer/train.py --init <converted file>`.

    python tools/convert_net.py net64b.nnue net/endgame-v2.nnue
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trainer"))
import train  # noqa: E402


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    with open(sys.argv[1], "rb") as f:
        data = f.read()
    if data[:4] != train.OLD_MAGIC:
        sys.exit("input is not an old-format (ENNU) network")
    q = train.expand_old(train.old_net_from_bytes(data))
    blob = train.net_to_bytes(q)
    with open(sys.argv[2], "wb") as f:
        f.write(blob)
    print(f"wrote {sys.argv[2]} ({len(blob)} bytes, {q['ft_b'].shape[0]} hidden units)")


if __name__ == "__main__":
    main()
