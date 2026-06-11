import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class DiffusionPoint:
    x: float
    y: float
    z: float
    time_sec: float
    pressure: float
    velocity_mag: float


@dataclass
class GroutDiffusionResult:
    injection_point_id: str
    elapsed_seconds: int
    predicted_radius_mm: float
    penetration_depth_mm: float
    pressure_kpa: float
    viscosity_pa_s: float
    porosity: float
    flow_rate_mls: float
    diffusion_front: List[Dict] = field(default_factory=list)
    particle_pathlines: List[Dict] = field(default_factory=list)
    volume_ml: float = 0.0
    concentration_distribution: Optional[np.ndarray] = None


class NewtonianSphericalDiffusion:
    def __init__(
        self,
        viscosity_pa_s: float = 0.25,
        porosity: float = 0.35,
        permeability_m2: float = 1e-12,
        wall_thickness_mm: float = 50.0,
    ):
        self.viscosity_pa_s = viscosity_pa_s
        self.porosity = porosity
        self.permeability_m2 = permeability_m2
        self.wall_thickness_m = wall_thickness_mm / 1000.0

    def _hagen_poiseuille_radius(
        self,
        delta_p_pa: float,
        time_s: float,
        radius_initial_m: float = 0.002,
    ) -> float:
        t = max(time_s, 1e-6)
        dp = max(delta_p_pa, 1e3)
        k = self.permeability_m2
        phi = self.porosity
        mu = self.viscosity_pa_s

        coefficient = (2.0 * k * dp) / (phi * mu)
        radius_squared = radius_initial_m ** 2 + coefficient * t
        radius_m = np.sqrt(max(radius_squared, radius_initial_m ** 2))

        radius_m = min(radius_m, self.wall_thickness_m * 2.5)

        return radius_m

    def _penetration_depth(
        self,
        delta_p_pa: float,
        time_s: float,
    ) -> float:
        t = max(time_s, 1e-6)
        dp = max(delta_p_pa, 1e3)
        k = self.permeability_m2
        phi = self.porosity
        mu = self.viscosity_pa_s

        depth_m = np.sqrt((2.0 * k * dp * t) / (phi * mu * 3.0))
        depth_m = min(depth_m, self.wall_thickness_m)

        return depth_m

    def _flow_rate(
        self,
        delta_p_pa: float,
        radius_m: float,
    ) -> float:
        dp = max(delta_p_pa, 1e3)
        k = self.permeability_m2
        mu = self.viscosity_pa_s
        thickness = self.wall_thickness_m

        q_m3s = (k * dp * 2.0 * np.pi * radius_m * thickness) / (mu * max(radius_m, 0.001))
        q_mls = q_m3s * 1e6

        return q_mls

    def _build_diffusion_front(
        self,
        center: Tuple[float, float, float],
        radius_m: float,
        n_points: int = 64,
    ) -> List[Dict]:
        cx, cy, cz = center
        points = []

        for i in range(n_points):
            theta = 2.0 * np.pi * i / n_points
            phi_angle = np.pi / 2.0
            jitter = 0.92 + 0.16 * np.random.rand()

            r = radius_m * jitter
            x = cx + r * np.sin(phi_angle) * np.cos(theta)
            y = cy + r * np.cos(phi_angle)
            z = cz + r * np.sin(phi_angle) * np.sin(theta)

            points.append({
                "x": round(float(x), 4),
                "y": round(float(y), 4),
                "z": round(float(z), 4),
                "concentration": round(float(jitter), 3),
            })

        return points

    def _build_particle_pathlines(
        self,
        center: Tuple[float, float, float],
        radius_m: float,
        depth_m: float,
        time_s: float,
        n_streamlines: int = 12,
        n_steps_per_line: int = 30,
    ) -> List[Dict]:
        cx, cy, cz = center
        streamlines = []

        for sl in range(n_streamlines):
            theta_0 = 2.0 * np.pi * sl / n_streamlines
            phi_0 = np.pi / 4.0 + np.random.uniform(-np.pi / 8, np.pi / 8)

            points = []
            for step in range(n_steps_per_line):
                t_frac = (step + 1) / n_steps_per_line
                r_current = radius_m * t_frac * (0.85 + 0.3 * np.random.rand())
                d_current = depth_m * t_frac

                theta = theta_0 + 0.15 * np.sin(t_frac * np.pi * 3)
                phi_a = phi_0 + 0.1 * np.cos(t_frac * np.pi * 2)

                x = cx + r_current * np.sin(phi_a) * np.cos(theta)
                y = cy + d_current * np.random.uniform(0.7, 1.0)
                z = cz + r_current * np.sin(phi_a) * np.sin(theta)

                vel = (r_current / max(time_s * t_frac, 0.001)) * 1000.0

                points.append({
                    "x": round(float(x), 4),
                    "y": round(float(y), 4),
                    "z": round(float(z), 4),
                    "t": round(float(time_s * t_frac), 1),
                    "v": round(float(vel), 3),
                    "c": round(float(1.0 - 0.5 * t_frac), 3),
                })

            streamlines.append({
                "streamline_id": f"SL-{sl:03d}",
                "start_theta": round(float(theta_0), 3),
                "points": points,
            })

        return streamlines

    def _compute_total_volume(
        self,
        radius_m: float,
        penetration_depth_m: float,
    ) -> float:
        r = radius_m
        h = penetration_depth_m
        volume_m3 = (2.0 / 3.0) * np.pi * (r ** 2) * h * self.porosity
        volume_ml = volume_m3 * 1e6
        return float(volume_ml)

    def predict_single_point(
        self,
        injection_point: Dict,
        elapsed_seconds: int,
        pressure_kpa: float,
    ) -> GroutDiffusionResult:
        center = (
            float(injection_point.get("x", 0.0)),
            float(injection_point.get("y", 0.0)),
            float(injection_point.get("z", 0.0)),
        )
        ip_id = injection_point.get("id", "IP-001")

        delta_p_pa = pressure_kpa * 1000.0
        t = float(max(elapsed_seconds, 1))

        radius_m = self._hagen_poiseuille_radius(delta_p_pa, t)
        penetration_m = self._penetration_depth(delta_p_pa, t)
        radius_mm = radius_m * 1000.0
        penetration_mm = penetration_m * 1000.0

        flow_rate = self._flow_rate(delta_p_pa, radius_m)
        volume_ml = self._compute_total_volume(radius_m, penetration_m)

        diffusion_front = self._build_diffusion_front(center, radius_m)
        pathlines = self._build_particle_pathlines(center, radius_m, penetration_m, t)

        return GroutDiffusionResult(
            injection_point_id=ip_id,
            elapsed_seconds=elapsed_seconds,
            predicted_radius_mm=round(float(radius_mm), 4),
            penetration_depth_mm=round(float(penetration_mm), 4),
            pressure_kpa=float(pressure_kpa),
            viscosity_pa_s=float(self.viscosity_pa_s),
            porosity=float(self.porosity),
            flow_rate_mls=round(float(flow_rate), 4),
            diffusion_front=diffusion_front,
            particle_pathlines=pathlines,
            volume_ml=round(float(volume_ml), 4),
        )

    def predict_multi_point(
        self,
        injection_points: List[Dict],
        elapsed_seconds: int,
        pressure_kpa: float,
        combine: bool = True,
    ) -> List[GroutDiffusionResult]:
        results = []
        for ip in injection_points:
            res = self.predict_single_point(ip, elapsed_seconds, pressure_kpa)
            results.append(res)

        return results


def assess_reinforcement_effectiveness(
    pre_freqs: List[float],
    post_freqs: List[float],
    pre_delamination_area: float,
    post_delamination_area: float,
    target_strength_mpa: float = 0.8,
) -> Dict:
    pre = np.array(pre_freqs, dtype=np.float64)
    post = np.array(post_freqs, dtype=np.float64)

    min_len = min(len(pre), len(post))
    pre = pre[:min_len]
    post = post[:min_len]

    freq_recovery_pcts = []
    for i in range(min_len):
        if pre[i] > 0:
            recovery = (post[i] - pre[i]) / pre[i] * 100.0
            freq_recovery_pcts.append(float(recovery))

    avg_freq_recovery = float(np.mean(freq_recovery_pcts)) if freq_recovery_pcts else 0.0

    if pre_delamination_area > 0:
        area_reduction_pct = (pre_delamination_area - post_delamination_area) / pre_delamination_area * 100.0
    else:
        area_reduction_pct = 100.0 if post_delamination_area == 0 else 0.0

    normalized_freq_recovery = min(max(avg_freq_recovery, 0.0), 30.0) / 30.0 * 100.0
    normalized_area_reduction = min(max(area_reduction_pct, 0.0), 100.0)

    strength_factor = min(target_strength_mpa, target_strength_mpa * normalized_freq_recovery / 100.0)
    bonding_strength = strength_factor * (0.8 + 0.2 * normalized_area_reduction / 100.0)

    overall_score = (
        normalized_freq_recovery * 0.45
        + normalized_area_reduction * 0.40
        + min(bonding_strength / target_strength_mpa * 100.0, 100.0) * 0.15
    )

    if overall_score >= 85:
        grade = "优秀"
        notes = "灌浆加固效果优异，壁画地仗层力学性能显著恢复"
    elif overall_score >= 70:
        grade = "良好"
        notes = "灌浆加固效果良好，大部分空鼓区域已得到有效填充"
    elif overall_score >= 55:
        grade = "合格"
        notes = "灌浆加固效果基本达标，建议对残余空鼓区域进行二次注浆"
    else:
        grade = "不合格"
        notes = "灌浆加固效果未达预期，需要重新评估注浆方案"

    return {
        "frequency_recovery_pct": round(avg_freq_recovery, 4),
        "delamination_area_reduction_pct": round(area_reduction_pct, 4),
        "bonding_strength_mpa": round(float(bonding_strength), 6),
        "overall_score": round(float(overall_score), 2),
        "grade": grade,
        "assessment_notes": notes,
        "per_mode_recovery_pct": [round(x, 4) for x in freq_recovery_pcts],
    }
