from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Pivot:
    index: int
    date: str
    price: float
    kind: str
    confirmed_at: int
    confirmed_date: str
    scale: str = "2ATR"


@dataclass
class Pattern:
    pattern_id: str
    code: str
    kind: str
    timeframe: str
    pattern_type: str
    direction: str
    state: str
    formed_at: str
    confirmed_at: Any
    key_points: Dict[str, Any] = field(default_factory=dict)
    neckline: Optional[Dict[str, float]] = None
    invalidation: Optional[float] = None
    targets: List[float] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    volume_ratio: Optional[float] = None
    volume_verified: bool = False
    rule_version: str = "caisen-v15"
    data_version: str = "unknown"
    parent_id: Optional[str] = None
    flags: List[str] = field(default_factory=list)
    execution_rules_verified: bool = False
    last_observed_date: Optional[str] = None
    confirmation_date: Optional[str] = None
    source_rule_version: str = "caisen-rules-v6"

    def to_dict(self):
        return asdict(self)
