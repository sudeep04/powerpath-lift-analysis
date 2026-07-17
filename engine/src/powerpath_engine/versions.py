"""Engine version constants recorded on every stored analysis.

Two independent version axes (a global constraint -- analyses must record
both so old results stay interpretable after the engine evolves):

* ``EXTRACTION_VERSION`` covers the signal-extraction half of the engine:
  decode, pose backends/scheduling, bar marker tracking, and calibration.
  Bump it when any of those changes in a way that would produce different
  extracted time series from the same video (new tracker thresholds, a
  different pose model mapping, changed calibration acceptance rules, ...).
  Stored analyses with an older extraction version were extracted with
  older signal code and should be re-extracted before being compared.

* ``RULES_VERSION`` covers the judgement half: segmentation, fault rules,
  made/missed criteria, scoring and (later) templates. It is OWNED by
  ``faults.py`` (bumped there when any rule's threshold or logic changes)
  and only re-exported here so the pipeline and API record both constants
  from one module without a second source of truth.
"""

from __future__ import annotations

from powerpath_engine.faults import RULES_VERSION

# Bump when decode / pose / bar tracking / calibration change what gets
# extracted from a video (see module docstring).
EXTRACTION_VERSION = 1

__all__ = ["EXTRACTION_VERSION", "RULES_VERSION"]
