"""
Open-Set Recognition (OSR) methods and inference modules.
"""

from src.methods.msp import MSPDetector
from src.methods.openmax import OpenMaxDetector
from src.methods.svdd import DeepSVDDDetector
from src.methods.dml import DMLDetector

__all__ = ["MSPDetector", "OpenMaxDetector", "DeepSVDDDetector", "DMLDetector"]
