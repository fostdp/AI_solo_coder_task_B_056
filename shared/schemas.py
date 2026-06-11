from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Optional, Any
from datetime import datetime


class Point3D(BaseModel):
    x: float
    y: float
    z: float


class VibrationDataBatch(BaseModel):
    timestamp: Optional[str] = None
    sensors: Dict[str, Dict[str, List[float]]]


class ThermalImageData(BaseModel):
    timestamp: Optional[str] = None
    camera_id: str
    temperature_matrix: Optional[List[List[float]]] = None
    thermal_bytes: Optional[bytes] = None
    image_path: Optional[str] = None
    max_temp: Optional[float] = None
    min_temp: Optional[float] = None
    avg_temp: Optional[float] = None
    hotspot_regions: Optional[List[Dict]] = None


class DataIngestResponse(BaseModel):
    success: bool
    message: str
    records_stored: int
    timestamp: str


class ModalAnalysisResponse(BaseModel):
    surface_id: str
    frequencies: List[float]
    damping_ratios: List[float]
    regions_count: int
    processing_ms: int
    regions: List[Dict]


class DiffusionSimulateRequest(BaseModel):
    task_id: str
    elapsed_seconds: Optional[int] = None


class DiffusionResponse(BaseModel):
    injection_point_id: str
    predicted_radius_mm: float
    penetration_depth_mm: float
    flow_rate_mls: float
    volume_ml: float
    elapsed_seconds: int
    diffusion_front_points: int
    streamlines_count: int


class EffectivenessAssessmentResponse(BaseModel):
    task_id: str
    surface_id: str
    pre_grout: Dict
    post_grout: Dict
    assessment: Dict


class AlertResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    alert_id: int
    cave_id: str
    surface_id: str
    alert_type: str
    severity: str
    message: str
    metrics: Optional[Dict] = None
    status: Optional[str] = None
    push_channels: Optional[Dict] = None
    created_at: Optional[datetime] = None


class GroutSimulateEvent(BaseModel):
    task_id: str
    surface_id: str
    injection_points: List[Dict]
    pressure_kpa: float
    elapsed_seconds: Optional[int] = None


class VibrationRawEvent(BaseModel):
    timestamp: str
    sensors: Dict[str, Dict[str, List[float]]]


class VibrationDenoisedEvent(BaseModel):
    surface_id: str
    timestamp: str
    sensors: Dict[str, Dict[str, List[float]]]


class ModalResultEvent(BaseModel):
    surface_id: str
    timestamp: str
    natural_frequencies: List[float]
    damping_ratios: List[float]
    model_order: int
    processing_ms: int


class DelaminationEvent(BaseModel):
    surface_id: str
    timestamp: str
    regions: List[Dict]
