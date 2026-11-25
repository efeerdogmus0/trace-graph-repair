"""Deterministic randomness that does not depend on platform or Python version.

`random.Random` helper methods (shuffle, sample, randrange) are not guaranteed to
produce identical sequences across Python releases. Data generated on a laptop must be
regenerable bit-for-bit on Colab, so every draw here is derived from a SHA-256 counter
stream using integer arithmetic only.
"""
from __future__ import annotations

import hashlib
from typing import Iterable, List, Sequence, TypeVar

T = TypeVar("T")

_SEP = "\x1f"


def _key(parts: Iterable[object]) -> bytes:
    return hashlib.sha256(_SEP.join(str(p) for p in parts).encode("utf-8")).digest()


def stable_hash(*parts: object, length: int = 16) -> str:
    """Hex digest of the given parts; stable across runs (unlike built-in hash())."""
    return _key(parts).hex()[:length]


class Rng:
    """SHA-256 counter-mode generator with a small, explicit API."""

    def __init__(self, *parts: object) -> None:
        self._k = _key(parts)
        self._ctr = 0
        self._buf = b""

    def child(self, *parts: object) -> "Rng":
        """Independent stream derived from this generator's key (not its position)."""
        return Rng(self._k.hex(), *parts)

    def _bytes(self, n: int) -> bytes:
        while len(self._buf) < n:
            self._buf += hashlib.sha256(self._k + self._ctr.to_bytes(8, "big")).digest()
            self._ctr += 1
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def bits(self, k: int) -> int:
        if k <= 0:
            return 0
        nbytes = (k + 7) // 8
        return int.from_bytes(self._bytes(nbytes), "big") >> (nbytes * 8 - k)

    def below(self, n: int) -> int:
        """Uniform integer in [0, n) by rejection sampling (no modulo bias)."""
        if n <= 0:
            raise ValueError("below() needs n >= 1")
        if n == 1:
            return 0
        k = (n - 1).bit_length()
        while True:
            v = self.bits(k)
            if v < n:
                return v

    def randint(self, a: int, b: int) -> int:
        """Uniform integer in [a, b] inclusive."""
        return a + self.below(b - a + 1)

    def random(self) -> float:
        """Float in [0, 1) with 53 random bits; exactly representable, so portable."""
        return self.bits(53) / float(1 << 53)

    def uniform(self, a: float, b: float) -> float:
        return a + (b - a) * self.random()

    def chance(self, p: float) -> bool:
        return self.random() < p

    def choice(self, seq: Sequence[T]) -> T:
        if not seq:
            raise ValueError("choice() from empty sequence")
        return seq[self.below(len(seq))]

    def shuffle(self, lst: List[T]) -> None:
        for i in range(len(lst) - 1, 0, -1):
            j = self.below(i + 1)
            lst[i], lst[j] = lst[j], lst[i]

    def shuffled(self, seq: Iterable[T]) -> List[T]:
        out = list(seq)
        self.shuffle(out)
        return out

    def sample(self, seq: Sequence[T], k: int) -> List[T]:
        return self.shuffled(seq)[: max(0, k)]

    def weighted_choice(self, items: Sequence[T], weights: Sequence[float]) -> T:
        total = float(sum(weights))
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        x = self.random() * total
        acc = 0.0
        for item, w in zip(items, weights):
            acc += w
            if x < acc:
                return item
        return items[-1]
