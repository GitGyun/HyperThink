"""Per-shard port selection for parallel vLLM engines.

Running N vLLM engines in parallel on one node makes each one init a
torch.distributed TCPStore. If they share a single inherited VLLM_PORT they bind
the same port -> DistNetworkError EADDRINUSE. Give each shard a distinct, free
port instead.
"""
import os
import socket


def _parse_reserved_ranges(text: str):
    """Parse /proc/sys/net/ipv4/ip_local_reserved_ports format ('a-b,c,d-e')
    into a list of (lo, hi) inclusive ranges. Returns [] on parse failure."""
    ranges = []
    for tok in (text or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            lo, _, hi = tok.partition("-")
            try:
                ranges.append((int(lo), int(hi)))
            except ValueError:
                return []
        else:
            try:
                p = int(tok)
                ranges.append((p, p))
            except ValueError:
                return []
    return ranges


def _pick_safe_base() -> int:
    """Choose a base port inside the kernel's ip_local_reserved_ports range
    so vLLM's fallback `bind(("", 0))` calls (which scan the ephemeral range
    *excluding* reserved ports) cannot collide with us. Falls back to 51000
    if the reserved range can't be parsed or is too small."""
    try:
        with open("/proc/sys/net/ipv4/ip_local_reserved_ports") as f:
            reserved = _parse_reserved_ranges(f.read())
    except OSError:
        reserved = []
    # Need a contiguous reserved span big enough for ~256 distinct ports.
    for lo, hi in reserved:
        if hi - lo >= 256:
            # Pick a base ~70% into the span to leave headroom on both sides.
            return lo + int((hi - lo) * 0.7)
    return 51000


_BASE = int(os.environ.get("VLLM_DIST_PORT_BASE", _pick_safe_base()))


def pick_free_port(shard_id: int, base: int = None, span: int = 16) -> int:
    """Return a distinct, currently-free TCP port for `shard_id`.

    The default base sits inside the kernel's `ip_local_reserved_ports` range
    so vLLM's secondary `bind(("", 0))` IPC ports, which the kernel assigns
    outside the reserved range, cannot collide with us. Override via
    `VLLM_DIST_PORT_BASE` env or `base=` arg.

    Tries a deterministic per-shard port (`base + shard_id*span`; spaced because
    vLLM uses a few consecutive ports), then falls back to an OS-assigned free
    port if that one is taken (e.g. a stale process from a crashed run).
    """
    if base is None:
        base = _BASE
    for cand in (base + shard_id * span, 0):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("", cand))
            return s.getsockname()[1]
        except OSError:
            continue
        finally:
            s.close()
    return base + shard_id * span
