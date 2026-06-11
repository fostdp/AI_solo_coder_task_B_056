from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc, func
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from backend.models.database import (
    VibrationRawData, ThermalImage, ModalAnalysisResult,
    DelaminationRegion, GroutingDiffusion,
)
from backend.models import get_db
from backend.schemas.models import (
    VibrationDataBatch, ThermalImageData, DataIngestResponse,
    ModalAnalysisResponse, DelaminationRegionResponse,
)
from backend.services.data_processing import (
    VibrationProcessingService, ThermalProcessingService, AlertDetectionService,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/data", tags=["数据接入与处理"])

vibration_service = VibrationProcessingService()
thermal_service = ThermalProcessingService()
alert_service = AlertDetectionService()


@router.post("/ingest/vibration", response_model=DataIngestResponse)
async def ingest_vibration(
    batch: VibrationDataBatch,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    try:
        count = await vibration_service.store_vibration_batch(db, batch)
        timestamp = datetime.utcnow().isoformat()

        surfaces_stmt = select(
            VibrationRawData.sensor_id
        ).where(VibrationRawData.time == (select(func.max(VibrationRawData.time)).scalar_subquery())).limit(1)

        return DataIngestResponse(
            success=True,
            message=f"成功存储 {count} 个传感器的振动数据",
            records_stored=count,
            timestamp=timestamp,
        )
    except Exception as e:
        logger.error(f"振动数据接入失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/thermal", response_model=DataIngestResponse)
async def ingest_thermal(
    image: ThermalImageData,
    db: AsyncSession = Depends(get_db),
):
    try:
        await thermal_service.store_thermal_image(db, image)
        return DataIngestResponse(
            success=True,
            message="热成像数据存储成功",
            records_stored=1,
            timestamp=datetime.utcnow().isoformat(),
        )
    except Exception as e:
        logger.error(f"热成像数据接入失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analysis/modal/{surface_id}", response_model=ModalAnalysisResponse)
async def run_modal_analysis(
    surface_id: str,
    db: AsyncSession = Depends(get_db),
):
    try:
        result = await vibration_service.run_modal_analysis(db, surface_id)
        if not result:
            raise HTTPException(status_code=404, detail=f"墙面 {surface_id} 无传感器数据")
        return ModalAnalysisResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"模态分析失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analysis/modal-all")
async def run_all_modal_analysis(
    db: AsyncSession = Depends(get_db),
):
    from backend.models.database import WallSurface

    stmt = select(WallSurface.surface_id)
    res = await db.execute(stmt)
    surfaces = [r[0] for r in res.all()]

    results = []
    for sid in surfaces:
        try:
            r = await vibration_service.run_modal_analysis(db, sid)
            if r:
                results.append({"surface_id": sid, "status": "success", "regions": r["regions_count"]})
            else:
                results.append({"surface_id": sid, "status": "skipped"})
        except Exception as e:
            results.append({"surface_id": sid, "status": "error", "error": str(e)})

    return {"total_surfaces": len(surfaces), "results": results}


@router.post("/alerts/check")
async def check_alerts(db: AsyncSession = Depends(get_db)):
    try:
        triggered = await alert_service.check_and_push_alerts(db)
        return {
            "success": True,
            "triggered_count": len(triggered),
            "alerts": triggered,
        }
    except Exception as e:
        logger.error(f"告警检查失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/delamination-regions", response_model=List[DelaminationRegionResponse])
async def get_delamination_regions(
    surface_id: Optional[str] = None,
    cave_id: Optional[str] = None,
    hours: int = Query(24, ge=1, le=24 * 30),
    active_only: bool = True,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(DelaminationRegion)
    if active_only:
        stmt = stmt.where(DelaminationRegion.is_active == True)
    if surface_id:
        stmt = stmt.where(DelaminationRegion.surface_id == surface_id)
    if cave_id:
        from backend.models.database import WallSurface
        subq = select(WallSurface.surface_id).where(WallSurface.cave_id == cave_id)
        stmt = stmt.where(DelaminationRegion.surface_id.in_(subq))

    stmt = stmt.where(DelaminationRegion.time >= datetime.utcnow() - timedelta(hours=hours))
    stmt = stmt.order_by(desc(DelaminationRegion.time))
    result = await db.execute(stmt)
    return result.scalars().all()


@router.get("/modal-results")
async def get_modal_results(
    surface_id: Optional[str] = None,
    days: int = Query(7, ge=1, le=90),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(ModalAnalysisResult)
    if surface_id:
        stmt = stmt.where(ModalAnalysisResult.surface_id == surface_id)
    stmt = stmt.where(ModalAnalysisResult.time >= datetime.utcnow() - timedelta(days=days))
    stmt = stmt.order_by(desc(ModalAnalysisResult.time)).limit(limit)
    result = await db.execute(stmt)
    rows = result.scalars().all()

    return [
        {
            "time": r.time.isoformat() if r.time else None,
            "surface_id": r.surface_id,
            "natural_frequencies": list(r.natural_frequencies) if r.natural_frequencies else [],
            "damping_ratios": list(r.damping_ratios) if r.damping_ratios else [],
            "ssi_model_order": r.ssi_model_order,
            "analyzed_sensors": list(r.analyzed_sensors) if r.analyzed_sensors else [],
            "processing_time_ms": r.processing_time_ms,
        }
        for r in rows
    ]


@router.get("/grouting/diffusion/{task_id}")
async def get_grouting_diffusion(
    task_id: str,
    latest_only: bool = True,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(GroutingDiffusion).where(GroutingDiffusion.task_id == task_id)
    if latest_only:
        subq = (
            select(func.max(GroutingDiffusion.time))
            .where(GroutingDiffusion.task_id == task_id)
            .scalar_subquery()
        )
        stmt = stmt.where(GroutingDiffusion.time == subq)
    stmt = stmt.order_by(desc(GroutingDiffusion.time))
    result = await db.execute(stmt)
    rows = result.scalars().all()

    return [
        {
            "time": r.time.isoformat() if r.time else None,
            "task_id": r.task_id,
            "injection_point_id": r.injection_point_id,
            "predicted_radius_mm": float(r.predicted_radius_mm) if r.predicted_radius_mm else None,
            "actual_radius_mm": float(r.actual_radius_mm) if r.actual_radius_mm else None,
            "penetration_depth_mm": float(r.penetration_depth_mm) if r.penetration_depth_mm else None,
            "pressure_kpa": float(r.pressure_kpa) if r.pressure_kpa else None,
            "flow_rate_mls": float(r.flow_rate_mls) if r.flow_rate_mls else None,
            "elapsed_seconds": r.elapsed_seconds,
            "diffusion_front": r.diffusion_front,
            "particle_pathlines": r.particle_pathlines,
            "viscosity_pa_s": float(r.viscosity_pa_s) if r.viscosity_pa_s else None,
            "porosity": float(r.porosity) if r.porosity else None,
        }
        for r in rows
    ]
