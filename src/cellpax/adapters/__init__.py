"""Thin downstream adapters over released CellPax contracts."""

from cellpax.adapters.trajan import (
    add_to_connectivity_table,
    add_to_synapse_table,
    annotation_frame,
    decorate_cells,
)

__all__ = [
    "add_to_connectivity_table",
    "add_to_synapse_table",
    "annotation_frame",
    "decorate_cells",
]
