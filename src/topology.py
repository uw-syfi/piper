from __future__ import annotations

from dataclasses import dataclass

from .schedule import load_schedule_info


@dataclass(frozen=True)
class Topology:
    """Parallel degrees of one training job and the global-rank layout they imply.

    dp: data-parallel degree.
    pp: pipeline-parallel degree.
    num_standby: spare ranks appended after the ``dp * pp`` trainer ranks.
    """

    dp: int
    pp: int = 1
    num_standby: int = 0

    def __post_init__(self) -> None:
        if self.dp < 1 or self.pp < 1 or self.num_standby < 0:
            raise ValueError(f"invalid topology: dp={self.dp} pp={self.pp} num_standby={self.num_standby}")

    @classmethod
    def from_schedule(cls, path: str, num_standby: int = 0) -> "Topology":
        """Read pp/dp degrees from the place directives of a schedule JSON."""
        info = load_schedule_info(path)
        return cls(dp=int(info["dp_degree"]), pp=int(info["pp_degree"]), num_standby=num_standby)

    @property
    def num_trainers(self) -> int:
        return self.dp * self.pp

    @property
    def world_size(self) -> int:
        return self.num_trainers + self.num_standby

    @property
    def standby_ranks(self) -> list[int]:
        return list(range(self.num_trainers, self.world_size))

    def slots(self) -> list[tuple[int, int]]:
        """All ``(dp, pp)`` trainer slots in global-rank order."""
        return [(dp, pp) for dp in range(self.dp) for pp in range(self.pp)]

    def rank(self, dp: int, pp: int) -> int:
        return pp + dp * self.pp

    def slot(self, rank: int) -> tuple[int, int] | None:
        """The ``(dp, pp)`` slot of a trainer rank, or None for a standby rank."""
        if rank >= self.num_trainers:
            return None
        return divmod(rank, self.pp)

    def pp_groups(self) -> list[list[int]]:
        """Rank lists of the pipeline groups, one per DP replica."""
        return [[self.rank(dp, pp) for pp in range(self.pp)] for dp in range(self.dp)]

    def dp_groups(self) -> list[list[int]]:
        """Rank lists of the DP/EP groups, one per pipeline rank."""
        return [[self.rank(dp, pp) for dp in range(self.dp)] for pp in range(self.pp)]

    def groups(self) -> dict[str, list[list[int]]]:
        """Explicit membership of every process group ``join_groups`` must create."""
        return {"dp": self.dp_groups(), "pp": self.pp_groups()}
