from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional, Any
from datetime import datetime


class Point3D(BaseModel):
    x: float
    y: float
    z: float


class CaveBase(BaseModel):
    cave_id: str
    cave_name: str
    dynasty: Optional[str] = None
    description: Optional[str] = None
    dimensions: Optional[Dict] = None


class CaveResponse(CaveBase):
    model_config = ConfigDict(from_attributes=True)
    location: Optional[Any] = None
    created_at: Optional[datetime] = None


class WallSurfaceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    surface_id: str
    cave_id: str
    wall_type: str
    area_sqm: Optional[float] = None
    bounding_box_3d: Optional[Dict] = None
    description: Optional[str] = None


class VibrationSensorResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    sensor_id: str
    cave_id: str
    surface_id: str
    location_3d: Dict
    sampling_rate_hz: Optional[int] = None
    sensitivity: Optional[float] = None
    model: Optional[str] = None
    status: Optional[str] = None


class ThermalCameraResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    camera_id: str
    cave_id: str
    surface_id: str
    location_3d: Dict
    resolution: Optional[str] = None
    temp_range_min: Optional[float] = None
    temp_range_max: Optional[float] = None
    model: Optional[str] = None
    status: Optional[str] = None


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


class ModalAnalysisResponse(BaseModel):
    surface_id: str
    frequencies: List[float]
    damping_ratios: List[float]
    regions_count: int
    processing_ms: int
    regions: List[Dict]


class DelaminationRegionResponse(BaseModel):
    time: datetime
    surface_id: str
    region_id: str
    bounding_polygon_3d: List[Dict]
    area_sqm: float
    depth_mm: Optional[float] = None
    severity_score: Optional[float] = None
    confidence: Optional[float] = None
    frequency_drop_pct: Optional[float] = None


class GroutingTaskResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    task_id: str
    cave_id: str
    surface_id: str
    material_type: Optional[str] = None
    injection_points: Optional[List[Dict]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_volume_ml: Optional[float] = None
    pressure_kpa: Optional[float] = None
    status: Optional[str] = None
    operator: Optional[str] = None
    notes: Optional[str] = None


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
    resolved_at: Optional[datetime] = None


class DataIngestResponse(BaseModel):
    success: bool
    message: str
    records_stored: int
    timestamp: str


class SystemStatusResponse(BaseModel):
    status: str
    vibration_sensors: int
    thermal_cameras: int
    active_alerts: int
    active_grouting_tasks: int
    last_processing_time: Optional[str] = None
