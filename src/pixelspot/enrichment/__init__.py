"""Perception enrichment: extra per-track signals beyond boxes and ids."""

from pixelspot.enrichment.face import FaceFinder
from pixelspot.enrichment.head_pose import HeadPoseEstimator

__all__ = ["FaceFinder", "HeadPoseEstimator"]
