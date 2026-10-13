"""predictive-audit: does a forecasting result beat 'tomorrow looks like today'?"""

from .audit import AuditReport, Check, audit

__all__ = ["audit", "AuditReport", "Check"]
__version__ = "0.1.0"
