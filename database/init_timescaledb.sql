-- ============================================================
-- 古代壁画地仗层剥离监测与灌浆加固效果评估系统
-- TimescaleDB 初始化脚本
-- ============================================================

-- 创建扩展
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;

-- ============================================================
-- 基础维度表
-- ============================================================

CREATE TABLE IF NOT EXISTS caves (
    cave_id VARCHAR(20) PRIMARY KEY,
    cave_name VARCHAR(100) NOT NULL,
    dynasty VARCHAR(50),
    location GEOGRAPHY(POINT, 4326),
    description TEXT,
    dimensions JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS wall_surfaces (
    surface_id VARCHAR(30) PRIMARY KEY,
    cave_id VARCHAR(20) REFERENCES caves(cave_id),
    wall_type VARCHAR(20) NOT NULL,
    area_sqm NUMERIC(10, 4),
    bounding_box_3d JSONB,
    description TEXT
);

CREATE TABLE IF NOT EXISTS vibration_sensors (
    sensor_id VARCHAR(30) PRIMARY KEY,
    cave_id VARCHAR(20) REFERENCES caves(cave_id),
    surface_id VARCHAR(30) REFERENCES wall_surfaces(surface_id),
    location_3d JSONB NOT NULL,
    sampling_rate_hz INTEGER DEFAULT 2000,
    sensitivity NUMERIC(10, 6),
    model VARCHAR(50),
    status VARCHAR(20) DEFAULT 'active',
    installed_at DATE
);

CREATE TABLE IF NOT EXISTS thermal_cameras (
    camera_id VARCHAR(30) PRIMARY KEY,
    cave_id VARCHAR(20) REFERENCES caves(cave_id),
    surface_id VARCHAR(30) REFERENCES wall_surfaces(surface_id),
    location_3d JSONB NOT NULL,
    resolution VARCHAR(20),
    temp_range_min NUMERIC(6, 2),
    temp_range_max NUMERIC(6, 2),
    model VARCHAR(50),
    status VARCHAR(20) DEFAULT 'active',
    installed_at DATE
);

CREATE TABLE IF NOT EXISTS grouting_tasks (
    task_id VARCHAR(40) PRIMARY KEY,
    cave_id VARCHAR(20) REFERENCES caves(cave_id),
    surface_id VARCHAR(30) REFERENCES wall_surfaces(surface_id),
    material_type VARCHAR(50) DEFAULT '烧结石粉+PS',
    injection_points JSONB,
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    total_volume_ml NUMERIC(12, 4),
    pressure_kpa NUMERIC(8, 4),
    status VARCHAR(20) DEFAULT 'pending',
    operator VARCHAR(50),
    notes TEXT
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id BIGSERIAL PRIMARY KEY,
    cave_id VARCHAR(20) REFERENCES caves(cave_id),
    surface_id VARCHAR(30) REFERENCES wall_surfaces(surface_id),
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    metrics JSONB,
    status VARCHAR(20) DEFAULT 'active',
    push_channels JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ
);

-- ============================================================
-- 时序数据表 (TimescaleDB Hypertable)
-- ============================================================

CREATE TABLE IF NOT EXISTS vibration_raw_data (
    time TIMESTAMPTZ NOT NULL,
    sensor_id VARCHAR(30) REFERENCES vibration_sensors(sensor_id),
    x_axis_accel DOUBLE PRECISION[],
    y_axis_accel DOUBLE PRECISION[],
    z_axis_accel DOUBLE PRECISION[],
    sample_count INTEGER,
    raw_data_hash VARCHAR(64),
    received_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('vibration_raw_data', 'time',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day');

CREATE INDEX IF NOT EXISTS idx_vibration_raw_sensor_time
    ON vibration_raw_data (sensor_id, time DESC);

CREATE TABLE IF NOT EXISTS thermal_images (
    time TIMESTAMPTZ NOT NULL,
    camera_id VARCHAR(30) REFERENCES thermal_cameras(camera_id),
    thermal_data BYTEA,
    temperature_matrix DOUBLE PRECISION[][],
    max_temp NUMERIC(8, 4),
    min_temp NUMERIC(8, 4),
    avg_temp NUMERIC(8, 4),
    hotspot_regions JSONB,
    image_path VARCHAR(255),
    received_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('thermal_images', 'time',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day');

CREATE INDEX IF NOT EXISTS idx_thermal_camera_time
    ON thermal_images (camera_id, time DESC);

CREATE TABLE IF NOT EXISTS modal_analysis_results (
    time TIMESTAMPTZ NOT NULL,
    surface_id VARCHAR(30) REFERENCES wall_surfaces(surface_id),
    natural_frequencies DOUBLE PRECISION[],
    damping_ratios DOUBLE PRECISION[],
    mode_shapes JSONB,
    ssi_model_order INTEGER,
    stability_diagram JSONB,
    analyzed_sensors VARCHAR(30)[],
    processing_time_ms INTEGER
);

SELECT create_hypertable('modal_analysis_results', 'time',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '7 days');

CREATE TABLE IF NOT EXISTS delamination_regions (
    time TIMESTAMPTZ NOT NULL,
    surface_id VARCHAR(30) REFERENCES wall_surfaces(surface_id),
    region_id VARCHAR(40),
    bounding_polygon_3d JSONB NOT NULL,
    area_sqm NUMERIC(10, 6) NOT NULL,
    depth_mm NUMERIC(8, 4),
    severity_score NUMERIC(5, 2),
    confidence NUMERIC(5, 4),
    frequency_drop_pct NUMERIC(8, 4),
    detection_method VARCHAR(30) DEFAULT 'SSI+Thermal',
    is_active BOOLEAN DEFAULT TRUE
);

SELECT create_hypertable('delamination_regions', 'time',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '7 days');

CREATE INDEX IF NOT EXISTS idx_delamination_surface_time
    ON delamination_regions (surface_id, time DESC);

CREATE TABLE IF NOT EXISTS grouting_diffusion (
    time TIMESTAMPTZ NOT NULL,
    task_id VARCHAR(40) REFERENCES grouting_tasks(task_id),
    injection_point_id VARCHAR(30),
    predicted_radius_mm NUMERIC(10, 4),
    actual_radius_mm NUMERIC(10, 4),
    penetration_depth_mm NUMERIC(10, 4),
    pressure_kpa NUMERIC(8, 4),
    viscosity_pa_s NUMERIC(10, 6),
    porosity NUMERIC(6, 4),
    flow_rate_mls NUMERIC(10, 4),
    diffusion_front JSONB,
    particle_pathlines JSONB,
    elapsed_seconds INTEGER,
    model_version VARCHAR(20) DEFAULT 'Newtonian_Spherical_v1'
);

SELECT create_hypertable('grouting_diffusion', 'time',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day');

CREATE TABLE IF NOT EXISTS reinforcement_effectiveness (
    time TIMESTAMPTZ NOT NULL,
    surface_id VARCHAR(30) REFERENCES wall_surfaces(surface_id),
    task_id VARCHAR(40) REFERENCES grouting_tasks(task_id),
    pre_grout_frequencies DOUBLE PRECISION[],
    post_grout_frequencies DOUBLE PRECISION[],
    frequency_recovery_pct NUMERIC(8, 4),
    delamination_area_reduction_pct NUMERIC(8, 4),
    bonding_strength_mpa NUMERIC(10, 6),
    overall_score NUMERIC(5, 2),
    assessment_notes TEXT
);

SELECT create_hypertable('reinforcement_effectiveness', 'time',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '30 days');

-- ============================================================
-- 连续聚合视图 (Materialized Views)
-- ============================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS delamination_daily_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time) AS bucket_day,
    surface_id,
    COUNT(DISTINCT region_id) AS active_region_count,
    SUM(area_sqm) AS total_delamination_area,
    AVG(severity_score) AS avg_severity,
    MAX(frequency_drop_pct) AS max_frequency_drop
FROM delamination_regions
WHERE is_active = TRUE
GROUP BY bucket_day, surface_id
WITH NO DATA;

CREATE MATERIALIZED VIEW IF NOT EXISTS vibration_modal_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket_hour,
    surface_id,
    AVG(natural_frequencies[1]) AS avg_fundamental_freq,
    AVG(damping_ratios[1]) AS avg_damping_ratio
FROM modal_analysis_results
GROUP BY bucket_hour, surface_id
WITH NO DATA;

-- ============================================================
-- 插入模拟基础数据
-- ============================================================

INSERT INTO caves (cave_id, cave_name, dynasty, location, description, dimensions) VALUES
('C096', '第96窟 (北大像窟)', '唐代', ST_SetSRID(ST_MakePoint(94.805630, 40.045260), 4326)::geography,
 '莫高窟标志性洞窟，内有35.5米高弥勒大佛', '{"length_m": 18.5, "width_m": 15.2, "height_m": 36.0}'),
('C257', '第257窟 (九色鹿窟)', '北魏', ST_SetSRID(ST_MakePoint(94.805680, 40.045290), 4326)::geography,
 '以九色鹿本生故事壁画闻名', '{"length_m": 12.0, "width_m": 9.5, "height_m": 7.5}'),
('C285', '第285窟', '西魏', ST_SetSRID(ST_MakePoint(94.805700, 40.045310), 4326)::geography,
 '有确切纪年的最早洞窟之一', '{"length_m": 10.5, "width_m": 8.0, "height_m": 6.0}')
ON CONFLICT DO NOTHING;

INSERT INTO wall_surfaces (surface_id, cave_id, wall_type, area_sqm, bounding_box_3d) VALUES
('C096-N', 'C096', 'north_wall', 547.20, '{"x_min": 0, "x_max": 15.2, "y_min": 0, "y_max": 36.0, "z_min": 18.5, "z_max": 18.5}'),
('C096-S', 'C096', 'south_wall', 547.20, '{"x_min": 0, "x_max": 15.2, "y_min": 0, "y_max": 36.0, "z_min": 0, "z_max": 0}'),
('C096-E', 'C096', 'east_wall', 666.00, '{"x_min": 0, "x_max": 0, "y_min": 0, "y_max": 36.0, "z_min": 0, "z_max": 18.5}'),
('C096-W', 'C096', 'west_wall', 666.00, '{"x_min": 15.2, "x_max": 15.2, "y_min": 0, "y_max": 36.0, "z_min": 0, "z_max": 18.5}'),
('C096-C', 'C096', 'ceiling', 281.20, '{"x_min": 0, "x_max": 15.2, "y_min": 36.0, "y_max": 36.0, "z_min": 0, "z_max": 18.5}'),
('C257-N', 'C257', 'north_wall', 71.25, '{"x_min": 0, "x_max": 9.5, "y_min": 0, "y_max": 7.5, "z_min": 12.0, "z_max": 12.0}'),
('C257-S', 'C257', 'south_wall', 71.25, '{"x_min": 0, "x_max": 9.5, "y_min": 0, "y_max": 7.5, "z_min": 0, "z_max": 0}'),
('C257-E', 'C257', 'east_wall', 90.00, '{"x_min": 0, "x_max": 0, "y_min": 0, "y_max": 7.5, "z_min": 0, "z_max": 12.0}'),
('C257-W', 'C257', 'west_wall', 90.00, '{"x_min": 9.5, "x_max": 9.5, "y_min": 0, "y_max": 7.5, "z_min": 0, "z_max": 12.0}'),
('C285-N', 'C285', 'north_wall', 63.00, '{"x_min": 0, "x_max": 8.0, "y_min": 0, "y_max": 6.0, "z_min": 10.5, "z_max": 10.5}'),
('C285-S', 'C285', 'south_wall', 63.00, '{"x_min": 0, "x_max": 8.0, "y_min": 0, "y_max": 6.0, "z_min": 0, "z_max": 0}')
ON CONFLICT DO NOTHING;

DO $$
DECLARE
    i INTEGER;
    j INTEGER;
    sensor_idx INTEGER := 0;
    camera_idx INTEGER := 0;
    surf RECORD;
    v_count INTEGER;
    h_count INTEGER;
    x_pos DOUBLE PRECISION;
    y_pos DOUBLE PRECISION;
    z_pos DOUBLE PRECISION;
BEGIN
    FOR surf IN SELECT * FROM wall_surfaces LOOP
        v_count := CASE
            WHEN surf.area_sqm > 500 THEN 10
            WHEN surf.area_sqm > 100 THEN 8
            WHEN surf.area_sqm > 60 THEN 5
            ELSE 3
        END;

        FOR i IN 0..(v_count-1) LOOP
            IF sensor_idx < 60 THEN
                x_pos := (random() * ((surf.bounding_box_3d->>'x_max')::DOUBLE PRECISION - (surf.bounding_box_3d->>'x_min')::DOUBLE PRECISION)) + (surf.bounding_box_3d->>'x_min')::DOUBLE PRECISION;
                y_pos := (random() * ((surf.bounding_box_3d->>'y_max')::DOUBLE PRECISION - (surf.bounding_box_3d->>'y_min')::DOUBLE PRECISION)) + (surf.bounding_box_3d->>'y_min')::DOUBLE PRECISION;
                z_pos := (random() * ((surf.bounding_box_3d->>'z_max')::DOUBLE PRECISION - (surf.bounding_box_3d->>'z_min')::DOUBLE PRECISION)) + (surf.bounding_box_3d->>'z_min')::DOUBLE PRECISION;

                INSERT INTO vibration_sensors (sensor_id, cave_id, surface_id, location_3d, sampling_rate_hz, sensitivity, model, installed_at)
                VALUES (
                    'VB-' || LPAD(sensor_idx::TEXT, 3, '0'),
                    surf.cave_id,
                    surf.surface_id,
                    jsonb_build_object('x', ROUND(x_pos, 3), 'y', ROUND(y_pos, 3), 'z', ROUND(z_pos, 3)),
                    2000,
                    0.00025,
                    'PCB-352C33',
                    '2025-03-15'::DATE
                ) ON CONFLICT DO NOTHING;
                sensor_idx := sensor_idx + 1;
            END IF;
        END LOOP;

        h_count := CASE
            WHEN surf.area_sqm > 500 THEN 4
            WHEN surf.area_sqm > 100 THEN 3
            WHEN surf.area_sqm > 60 THEN 2
            ELSE 1
        END;

        FOR j IN 0..(h_count-1) LOOP
            IF camera_idx < 20 THEN
                x_pos := (random() * ((surf.bounding_box_3d->>'x_max')::DOUBLE PRECISION - (surf.bounding_box_3d->>'x_min')::DOUBLE PRECISION)) + (surf.bounding_box_3d->>'x_min')::DOUBLE PRECISION;
                y_pos := (random() * ((surf.bounding_box_3d->>'y_max')::DOUBLE PRECISION - (surf.bounding_box_3d->>'y_min')::DOUBLE PRECISION)) + (surf.bounding_box_3d->>'y_min')::DOUBLE PRECISION;
                z_pos := (random() * ((surf.bounding_box_3d->>'z_max')::DOUBLE PRECISION - (surf.bounding_box_3d->>'z_min')::DOUBLE PRECISION)) + (surf.bounding_box_3d->>'z_min')::DOUBLE PRECISION + 0.5;

                INSERT INTO thermal_cameras (camera_id, cave_id, surface_id, location_3d, resolution, temp_range_min, temp_range_max, model, installed_at)
                VALUES (
                    'TH-' || LPAD(camera_idx::TEXT, 3, '0'),
                    surf.cave_id,
                    surf.surface_id,
                    jsonb_build_object('x', ROUND(x_pos, 3), 'y', ROUND(y_pos, 3), 'z', ROUND(z_pos, 3)),
                    '640x480',
                    -20.0,
                    150.0,
                    'FLIR-T1040sc',
                    '2025-03-15'::DATE
                ) ON CONFLICT DO NOTHING;
                camera_idx := camera_idx + 1;
            END IF;
        END LOOP;
    END LOOP;
END $$;

INSERT INTO grouting_tasks (task_id, cave_id, surface_id, material_type, injection_points, start_time, status, operator) VALUES
('GRK-2026-001', 'C096', 'C096-N', '烧结石粉+PS',
 '[{"id": "IP-01", "x": 5.2, "y": 8.5, "z": 18.5}, {"id": "IP-02", "x": 10.5, "y": 12.0, "z": 18.5}, {"id": "IP-03", "x": 7.8, "y": 18.3, "z": 18.5}]',
 '2026-06-01 09:00:00+08', 'in_progress', '王修复师'),
('GRK-2026-002', 'C257', 'C257-N', '烧结石粉+PS',
 '[{"id": "IP-01", "x": 3.2, "y": 3.5, "z": 12.0}, {"id": "IP-02", "x": 6.0, "y": 4.2, "z": 12.0}]',
 '2026-06-10 14:00:00+08', 'completed', '李修复师')
ON CONFLICT DO NOTHING;
