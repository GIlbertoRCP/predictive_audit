"""predictive-audit: does a forecasting result beat 'tomorrow looks like today'?"""

from . import baselines, metrics, report
from ._backend import name as backend
from .audit import SCHEMA_VERSION, AuditReport, BatchAuditReport, Check, audit, batch_audit
from .reshape import wide_to_long

__all__ = [
    "AuditReport",
    "BatchAuditReport",
    "Check",
    "SCHEMA_VERSION",
    "audit",
    "backend",
    "baselines",
    "batch_audit",
    "metrics",
    "report",
    "wide_to_long",
]
__version__ = "0.2.0"
