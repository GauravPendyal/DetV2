"""Public agent surface — mirrors V1's ``DetectorAgent().detect(...)``.

The pipeline (and its models) is built ONCE per agent instance and reused across
calls, so nothing is reloaded per inference. ``detect`` returns the extended dict
(canonical six fields + claim detail); ``detect_output`` returns the rich pydantic
object; ``detect_canonical`` returns strictly the six fields.

Default detector (Stage 7): a bare ``DetectorAgent()`` wires the calibrated Stage-2
DeBERTa encoder via the existing :func:`build_stage7_agent` factory, loading the
on-disk artifacts resolved relative to the DetV2 repo root (not the caller's CWD).
The heavy encoder loads lazily on the first inference, so construction stays cheap.

Failure policy for the default path:
  * artifacts **missing** -> honest fall back to the Stage-0 pipeline, which
    self-identifies as ``detector_v2_stage0_heuristic`` (never disguised as Stage 7);
  * artifacts **present but fail to load** -> raise loudly, so a deployment can
    never silently run the wrong detector.

Explicit injection is always honored: ``DetectorAgent(pipeline=...)`` uses the given
pipeline unchanged, and ``DetectorAgent(config=...)`` honors the supplied config
(e.g. ``stage="stage0_heuristic"``) instead of loading Stage 7.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from .adapter import to_canonical_dict, to_extended_dict
from .config import DetectorV2Config
from .pipeline import DetectorV2Pipeline
from .schemas import DetectorV2Output

# Default Stage-7 artifacts, resolved relative to the DetV2 repo root (the parent
# of this package) so a bare DetectorAgent() finds them regardless of caller CWD.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENCODER_DIR = os.path.join(_REPO_ROOT, "models", "stage2_encoder")
_CALIBRATOR = os.path.join(_REPO_ROOT, "models", "stage7_calibration", "calibrator.joblib")


def _default_stage7_pipeline() -> DetectorV2Pipeline:
    """Build the calibrated Stage-2 encoder + Stage-7 pipeline from disk.

    Missing artifacts -> Stage-0 fallback (self-labeled, honest). Artifacts that
    exist but fail to load -> RuntimeError (never a silent wrong-detector run).
    """
    encoder_present = os.path.isdir(_ENCODER_DIR)
    calibrator_present = os.path.isfile(_CALIBRATOR)
    if not (encoder_present and calibrator_present):
        # Genuinely absent -> honest Stage-0 default (reports its own model_source).
        return DetectorV2Pipeline()
    try:
        from .calibration.build import build_stage7_agent
        from .calibration.calibrator import ProbabilityCalibrator

        calibrator = ProbabilityCalibrator.load(_CALIBRATOR)
        # Factory returns a DetectorAgent(pipeline=...); reuse just its pipeline.
        return build_stage7_agent(_ENCODER_DIR, calibrator).pipeline
    except Exception as exc:  # noqa: BLE001 - present-but-broken must fail loudly
        raise RuntimeError(
            "Stage-7 artifacts are present but failed to load "
            f"(encoder={_ENCODER_DIR}, calibrator={_CALIBRATOR}): "
            f"{type(exc).__name__}: {exc}. Refusing to silently fall back to Stage 0."
        ) from exc


class DetectorAgent:
    def __init__(self, config: Optional[DetectorV2Config] = None, pipeline: Optional[DetectorV2Pipeline] = None):
        if pipeline is not None:
            self.pipeline = pipeline                              # explicit DI, unchanged
        elif config is not None:
            self.pipeline = DetectorV2Pipeline(config=config)     # honor supplied config
        else:
            self.pipeline = _default_stage7_pipeline()            # bare agent -> Stage 7

    def detect_output(
        self, user_query: str, llm_response: str, context: Optional[str] = None,
        domain: str = "general", query_id: Optional[str] = None,
    ) -> DetectorV2Output:
        return self.pipeline.detect(user_query, llm_response, context, domain, query_id)

    def detect(self, user_query: str, llm_response: str, context: Optional[str] = None,
               domain: str = "general", query_id: Optional[str] = None) -> Dict[str, Any]:
        return to_extended_dict(self.detect_output(user_query, llm_response, context, domain, query_id))

    def detect_canonical(self, user_query: str, llm_response: str, context: Optional[str] = None,
                         domain: str = "general", query_id: Optional[str] = None) -> Dict[str, Any]:
        return to_canonical_dict(self.detect_output(user_query, llm_response, context, domain, query_id))
