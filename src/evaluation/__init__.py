"""
Evaluation and visualization modules.
"""

from src.evaluation.metrics import (
    compute_closed_set_metrics,
    compute_osr_binary_metrics,
    compute_ece,
    compute_oscr,
)
from src.evaluation.visualizations import (
    plot_roc_comparison,
    plot_unknown_detection_per_class,
    plot_tsne_embeddings,
    plot_master_dashboard,
)

__all__ = [
    "compute_closed_set_metrics",
    "compute_osr_binary_metrics",
    "compute_ece",
    "compute_oscr",
    "plot_roc_comparison",
    "plot_unknown_detection_per_class",
    "plot_tsne_embeddings",
    "plot_master_dashboard",
]
