"""predictive-audit: does a forecasting result beat 'tomorrow looks like today'?"""

from .audit import AuditReport, Check, audit
from .reshape import wide_to_long

__all__ = ["audit", "AuditReport", "Check", "wide_to_long"]
__version__ = "0.1.0"
