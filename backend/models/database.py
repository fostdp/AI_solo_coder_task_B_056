from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Boolean, Text,
    ForeignKey, Index, BigInteger, Numeric, ARRAY, LargeBinary
)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, DOUBLE_PRECISION, GEOGRAPHY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from geoalchemy2 import Geography

Base = declarative_base()


class Cave(Base):
    __tablename__ = "caves"
    cave_id = Column(String(20), primary_key=True)
    cave_name = Column(String(100), nullable=False)
    dynasty = Column(String(50))
    location = Column(Geography(geometry_type="POINT", srid=4326))
    description = Column(Text)
    dimensions = Column(JSONB)
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow)

    wall_surfaces = relationship("WallSurface", back_populates="cave")


class WallSurface(Base):
    __tablename__ = "wall_surfaces"
    surface_id = Column(String(30), primary_key=True)
    cave_id = Column(String(20), ForeignKey("caves.cave_id"))
    wall_type = Column(String(20), nullable=False)
    area_sqm = Column(Numeric(10, 4))
    bounding_box_3d = Column(JSONB)
    description = Column(Text)

    cave = relationship("Cave", back_populates="wall_surfaces")
    vibration_sensors = relationship("VibrationSensor", back_populates="surface")
    thermal_cameras = relationship("ThermalCamera", back_populates="surface")


class VibrationSensor(Base):
    __tablename__ = "vibration_sensors"
    sensor_id = Column(String(30), primary_key=True)
    cave_id = Column(String(20), ForeignKey("caves.cave_id"))
    surface_id = Column(String(30), ForeignKey("wall_surfaces.surface_id"))
    location_3d = Column(JSONB, nullable=False)
    sampling_rate_hz = Column(Integer, default=2000)
    sensitivity = Column(Numeric(10, 6))
    model = Column(String(50))
    status = Column(String(20), default="active")
    installed_at = Column(DateTime)

    surface = relationship("WallSurface", back_populates="vibration_sensors")


class ThermalCamera(Base):
    __tablename__ = "thermal_cameras"
    camera_id = Column(String(30), primary_key=True)
    cave_id = Column(String(20), ForeignKey("caves.cave_id"))
    surface_id = Column(String(30), ForeignKey("wall_surfaces.surface_id"))
    location_3d = Column(JSONB, nullable=False)
    resolution = Column(String(20))
    temp_range_min = Column(Numeric(6, 2))
    temp_range_max = Column(Numeric(6, 2))
    model = Column(String(50))
    status = Column(String(20), default="active")
    installed_at = Column(DateTime)

    surface = relationship("WallSurface", back_populates="thermal_cameras")


class GroutingTask(Base):
    __tablename__ = "grouting_tasks"
    task_id = Column(String(40), primary_key=True)
    cave_id = Column(String(20), ForeignKey("caves.cave_id"))
    surface_id = Column(String(30), ForeignKey("wall_surfaces.surface_id"))
    material_type = Column(String(50), default="烧结石粉+PS")
    injection_points = Column(JSONB)
    start_time = Column(TIMESTAMP(timezone=True))
    end_time = Column(TIMESTAMP(timezone=True))
    total_volume_ml = Column(Numeric(12, 4))
    pressure_kpa = Column(Numeric(8, 4))
    status = Column(String(20), default="pending")
    operator = Column(String(50))
    notes = Column(Text)


class Alert(Base):
    __tablename__ = "alerts"
    alert_id = Column(BigInteger, primary_key=True, autoincrement=True)
    cave_id = Column(String(20), ForeignKey("caves.cave_id"))
    surface_id = Column(String(30), ForeignKey("wall_surfaces.surface_id"))
    alert_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)
    message = Column(Text, nullable=False)
    metrics = Column(JSONB)
    status = Column(String(20), default="active")
    push_channels = Column(JSONB)
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow)
    resolved_at = Column(TIMESTAMP(timezone=True))


class VibrationRawData(Base):
    __tablename__ = "vibration_raw_data"
    time = Column(TIMESTAMP(timezone=True), primary_key=True)
    sensor_id = Column(String(30), ForeignKey("vibration_sensors.sensor_id"), primary_key=True)
    x_axis_accel = Column(ARRAY(DOUBLE_PRECISION))
    y_axis_accel = Column(ARRAY(DOUBLE_PRECISION))
    z_axis_accel = Column(ARRAY(DOUBLE_PRECISION))
    sample_count = Column(Integer)
    raw_data_hash = Column(String(64))
    received_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow)


class ThermalImage(Base):
    __tablename__ = "thermal_images"
    time = Column(TIMESTAMP(timezone=True), primary_key=True)
    camera_id = Column(String(30), ForeignKey("thermal_cameras.camera_id"), primary_key=True)
    thermal_data = Column(LargeBinary)
    temperature_matrix = Column(ARRAY(ARRAY(DOUBLE_PRECISION)))
    max_temp = Column(Numeric(8, 4))
    min_temp = Column(Numeric(8, 4))
    avg_temp = Column(Numeric(8, 4))
    hotspot_regions = Column(JSONB)
    image_path = Column(String(255))
    received_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow)


class ModalAnalysisResult(Base):
    __tablename__ = "modal_analysis_results"
    time = Column(TIMESTAMP(timezone=True), primary_key=True)
    surface_id = Column(String(30), ForeignKey("wall_surfaces.surface_id"), primary_key=True)
    natural_frequencies = Column(ARRAY(DOUBLE_PRECISION))
    damping_ratios = Column(ARRAY(DOUBLE_PRECISION))
    mode_shapes = Column(JSONB)
    ssi_model_order = Column(Integer)
    stability_diagram = Column(JSONB)
    analyzed_sensors = Column(ARRAY(String(30)))
    processing_time_ms = Column(Integer)


class DelaminationRegion(Base):
    __tablename__ = "delamination_regions"
    time = Column(TIMESTAMP(timezone=True), primary_key=True)
    surface_id = Column(String(30), ForeignKey("wall_surfaces.surface_id"), primary_key=True)
    region_id = Column(String(40), primary_key=True)
    bounding_polygon_3d = Column(JSONB, nullable=False)
    area_sqm = Column(Numeric(10, 6), nullable=False)
    depth_mm = Column(Numeric(8, 4))
    severity_score = Column(Numeric(5, 2))
    confidence = Column(Numeric(5, 4))
    frequency_drop_pct = Column(Numeric(8, 4))
    detection_method = Column(String(30), default="SSI+Thermal")
    is_active = Column(Boolean, default=True)


class GroutingDiffusion(Base):
    __tablename__ = "grouting_diffusion"
    time = Column(TIMESTAMP(timezone=True), primary_key=True)
    task_id = Column(String(40), ForeignKey("grouting_tasks.task_id"), primary_key=True)
    injection_point_id = Column(String(30), primary_key=True)
    predicted_radius_mm = Column(Numeric(10, 4))
    actual_radius_mm = Column(Numeric(10, 4))
    penetration_depth_mm = Column(Numeric(10, 4))
    pressure_kpa = Column(Numeric(8, 4))
    viscosity_pa_s = Column(Numeric(10, 6))
    porosity = Column(Numeric(6, 4))
    flow_rate_mls = Column(Numeric(10, 4))
    diffusion_front = Column(JSONB)
    particle_pathlines = Column(JSONB)
    elapsed_seconds = Column(Integer)
    model_version = Column(String(20), default="Newtonian_Spherical_v1")


class ReinforcementEffectiveness(Base):
    __tablename__ = "reinforcement_effectiveness"
    time = Column(TIMESTAMP(timezone=True), primary_key=True)
    surface_id = Column(String(30), ForeignKey("wall_surfaces.surface_id"), primary_key=True)
    task_id = Column(String(40), ForeignKey("grouting_tasks.task_id"), primary_key=True)
    pre_grout_frequencies = Column(ARRAY(DOUBLE_PRECISION))
    post_grout_frequencies = Column(ARRAY(DOUBLE_PRECISION))
    frequency_recovery_pct = Column(Numeric(8, 4))
    delamination_area_reduction_pct = Column(Numeric(8, 4))
    bonding_strength_mpa = Column(Numeric(10, 6))
    overall_score = Column(Numeric(5, 2))
    assessment_notes = Column(Text)
