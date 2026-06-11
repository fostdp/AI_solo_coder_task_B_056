import sys
import os
import json
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.algorithms.modal.wavelet_denoise import (
    WaveletThresholdDenoiser,
    preprocess_vibration_signal,
)
from backend.algorithms.ssi_modal import (
    StochasticSubspaceIdentification,
    detect_delamination_regions,
)
from backend.algorithms.grouting_diffusion import (
    NewtonianSphericalDiffusion,
    assess_reinforcement_effectiveness,
)
from shared.config import SharedSettings
from shared.schemas import VibrationDataBatch, ThermalImageData


class TestWaveletDenoise:
    def test_denoise_removes_high_freq_noise(self):
        np.random.seed(42)
        fs = 2000.0
        t = np.arange(5000) / fs
        clean = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 25 * t)
        noise = 0.3 * np.random.randn(len(t))
        noisy = clean + noise

        denoiser = WaveletThresholdDenoiser(wavelet="db8", level=5, mode="soft", threshold_method="rigrsure")
        denoised = denoiser.denoise(noisy)

        err_noisy = np.mean((noisy - clean) ** 2)
        err_denoised = np.mean((denoised - clean) ** 2)
        assert err_denoised < err_noisy, f"去噪后误差应小于原始误差: {err_denoised:.4f} >= {err_noisy:.4f}"

    def test_denoise_multichannel(self):
        np.random.seed(0)
        n_ch, n_s = 6, 3000
        data = np.random.randn(n_ch, n_s) * 0.1
        for ch in range(n_ch):
            data[ch] += np.sin(2 * np.pi * (5 + ch * 3) * np.arange(n_s) / 2000)

        denoiser = WaveletThresholdDenoiser()
        result = denoiser.denoise_multichannel(data)
        assert result.shape == data.shape

    def test_preprocess_pipeline(self):
        np.random.seed(7)
        signal = np.random.randn(2000) * 0.5 + np.sin(2 * np.pi * 15 * np.arange(2000) / 2000)
        processed = preprocess_vibration_signal(signal, fs=2000.0, detrend=True)
        assert len(processed) == len(signal)

    def test_soft_vs_hard(self):
        np.random.seed(1)
        sig = np.sin(2 * np.pi * 10 * np.arange(1000) / 2000) + 0.2 * np.random.randn(1000)
        soft = WaveletThresholdDenoiser(mode="soft").denoise(sig)
        hard = WaveletThresholdDenoiser(mode="hard").denoise(sig)
        assert len(soft) == len(hard) == len(sig)

    def test_threshold_methods(self):
        np.random.seed(2)
        sig = np.sin(2 * np.pi * 8 * np.arange(2000) / 2000) + 0.3 * np.random.randn(2000)
        for method in ["universal", "rigrsure", "sqtwolog"]:
            denoiser = WaveletThresholdDenoiser(threshold_method=method)
            result = denoiser.denoise(sig)
            assert len(result) == len(sig)


class TestSSIModal:
    def _make_vibration_data(self, n_sensors=5, n_samples=2000, fs=2000.0):
        np.random.seed(100)
        data = {}
        freqs_true = [5.0, 12.0, 25.0]
        t = np.arange(n_samples) / fs
        for i in range(n_sensors):
            axes = {}
            for axis in ["x", "y", "z"]:
                sig = np.zeros(n_samples)
                for f in freqs_true:
                    sig += (0.5 + np.random.rand() * 0.5) * np.sin(2 * np.pi * f * t + np.random.rand() * np.pi)
                sig += 0.05 * np.random.randn(n_samples)
                axes[axis] = sig.tolist()
            data[f"S-{i:03d}"] = axes
        return data

    def test_identify_returns_frequencies(self):
        data = self._make_vibration_data()
        ssi = StochasticSubspaceIdentification(
            fs=2000.0, order_min=10, order_max=30,
            wavelet_denoise=True,
        )
        result = ssi.identify(data)
        assert len(result.frequencies) > 0
        assert len(result.damping_ratios) > 0
        assert len(result.frequencies) == len(result.damping_ratios)
        assert np.all(result.frequencies > 0)
        assert np.all(result.damping_ratios >= 0)

    def test_ssi_with_denoise_vs_without(self):
        data = self._make_vibration_data()
        ssi_dn = StochasticSubspaceIdentification(fs=2000.0, order_min=10, order_max=25, wavelet_denoise=True)
        ssi_nd = StochasticSubspaceIdentification(fs=2000.0, order_min=10, order_max=25, wavelet_denoise=False)
        r_dn = ssi_dn.identify(data)
        r_nd = ssi_nd.identify(data)
        assert len(r_dn.frequencies) > 0
        assert len(r_nd.frequencies) > 0

    def test_detect_delamination(self):
        data = self._make_vibration_data()
        ssi = StochasticSubspaceIdentification(fs=2000.0, order_min=10, order_max=25, wavelet_denoise=True)
        result = ssi.identify(data)
        baseline = result.frequencies * 1.08
        positions = [{"x": i, "y": 0, "z": 0} for i in range(5)]
        regions = detect_delamination_regions(result, baseline, positions)
        assert isinstance(regions, list)


class TestGroutDiffusion:
    def test_single_point_prediction(self):
        model = NewtonianSphericalDiffusion(
            viscosity_pa_s=0.25,
            porosity=0.35,
            permeability_m2=1e-12,
            wall_thickness_mm=50.0,
        )
        ip = {"id": "IP-01", "x": 5.0, "y": 10.0, "z": 8.0}
        result = model.predict_single_point(ip, elapsed_seconds=3600, pressure_kpa=50.0)
        assert result.predicted_radius_mm > 0
        assert result.penetration_depth_mm > 0
        assert result.flow_rate_mls >= 0
        assert len(result.diffusion_front) > 0
        assert len(result.particle_pathlines) > 0

    def test_multi_point_prediction(self):
        model = NewtonianSphericalDiffusion()
        ips = [
            {"id": f"IP-{i:02d}", "x": float(i * 2), "y": 5.0, "z": 5.0}
            for i in range(3)
        ]
        results = model.predict_multi_point(ips, elapsed_seconds=1800, pressure_kpa=60.0)
        assert len(results) == 3
        for r in results:
            assert r.predicted_radius_mm > 0

    def test_diffusion_increases_with_time(self):
        model = NewtonianSphericalDiffusion()
        ip = {"id": "IP-T", "x": 1.0, "y": 1.0, "z": 1.0}
        r1 = model.predict_single_point(ip, elapsed_seconds=600, pressure_kpa=50.0)
        r2 = model.predict_single_point(ip, elapsed_seconds=3600, pressure_kpa=50.0)
        assert r2.predicted_radius_mm > r1.predicted_radius_mm

    def test_assess_effectiveness(self):
        pre_freqs = [8.5, 22.0, 41.5, 60.2, 85.0]
        post_freqs = [9.5, 24.0, 44.0, 63.5, 89.0]
        assessment = assess_reinforcement_effectiveness(pre_freqs, post_freqs, 0.85, 0.15)
        assert "overall_score" in assessment
        assert "grade" in assessment
        assert assessment["overall_score"] > 0


class TestSharedConfig:
    def test_settings_defaults(self):
        s = SharedSettings()
        assert s.SSI_MODEL_ORDER_MIN == 10
        assert s.SSI_MODEL_ORDER_MAX == 50
        assert s.WAVELET_NAME == "db8"
        assert s.REDIS_URL.startswith("redis://")
        assert s.ALERT_AREA_INCREASE_PCT == 10.0
        assert s.ALERT_FREQ_DROP_PCT == 5.0

    def test_yaml_load(self):
        s = SharedSettings()
        yaml_path = os.path.join(os.path.dirname(__file__), "..", "params", "default.yaml")
        s.load_params_from_yaml(yaml_path)
        assert s.SSI_MODEL_ORDER_MIN >= 1


class TestSchemas:
    def test_vibration_batch(self):
        batch = VibrationDataBatch(
            timestamp="2026-01-01T00:00:00Z",
            sensors={"S-001": {"x": [0.1, 0.2], "y": [0.3, 0.4], "z": [0.5, 0.6]}},
        )
        assert batch.sensors["S-001"]["x"] == [0.1, 0.2]

    def test_thermal_image(self):
        img = ThermalImageData(
            camera_id="TH-001",
            max_temp=25.5,
            min_temp=18.2,
            avg_temp=21.3,
        )
        assert img.camera_id == "TH-001"
        assert img.max_temp == 25.5


class TestRedisStreamFormat:
    def test_message_serialization(self):
        data = {
            "type": "vibration_raw",
            "surface_id": "C096-N",
            "timestamp": "2026-01-01T00:00:00Z",
            "sensors": json.dumps({"S-001": {"x": [0.1]}}),
        }
        for k, v in data.items():
            assert isinstance(v, str), f"Redis Stream value must be str: {k}={type(v)}"

    def test_delamination_event_format(self):
        event = {
            "type": "delamination_detected",
            "surface_id": "C096-N",
            "timestamp": "2026-01-01T00:00:00Z",
            "n_regions": "3",
            "regions": json.dumps([{"region_id": "DEL-1", "area_sqm": 0.5}]),
        }
        parsed = json.loads(event["regions"])
        assert len(parsed) == 1
        assert parsed[0]["region_id"] == "DEL-1"


class TestEndToEndPipeline:
    def test_vibration_to_modal_pipeline(self):
        np.random.seed(42)
        fs = 2000.0
        n_samples = 2000
        t = np.arange(n_samples) / fs
        freqs_true = [8.0, 20.0]
        sensors = {}
        for i in range(4):
            axes = {}
            for axis in ["x", "y", "z"]:
                sig = sum(
                    (0.5 + np.random.rand() * 0.3) * np.sin(2 * np.pi * f * t + np.random.rand() * 2 * np.pi)
                    for f in freqs_true
                )
                sig += 0.05 * np.random.randn(n_samples)
                axes[axis] = sig.tolist()
            sensors[f"VB-{i:03d}"] = axes

        denoiser = WaveletThresholdDenoiser(wavelet="db8", level=4, mode="soft", threshold_method="rigrsure")
        for sid, axes in sensors.items():
            for axis in ["x", "y", "z"]:
                raw = np.array(axes[axis])
                denoised = denoiser.denoise(raw - np.mean(raw))
                sensors[sid][axis] = denoised.tolist()

        ssi = StochasticSubspaceIdentification(fs=fs, order_min=6, order_max=20, wavelet_denoise=False)
        result = ssi.identify(sensors)
        assert len(result.frequencies) > 0

    def test_grout_pipeline(self):
        model = NewtonianSphericalDiffusion(
            viscosity_pa_s=0.25, porosity=0.35, permeability_m2=1e-12, wall_thickness_mm=50.0,
        )
        ips = [{"id": "IP-01", "x": 5.0, "y": 10.0, "z": 8.0}]
        results = model.predict_multi_point(ips, elapsed_seconds=3600, pressure_kpa=50.0)
        assert len(results) == 1

        pre_freqs = [8.0, 20.0, 40.0]
        post_freqs = [f * 1.1 for f in pre_freqs]
        assessment = assess_reinforcement_effectiveness(pre_freqs, post_freqs, 0.8, 0.2)
        assert assessment["overall_score"] > 50


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
