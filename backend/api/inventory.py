from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, desc
from typing import List, Optional
from datetime import datetime, timedelta

from backend.models.database import (
    Cave, WallSurface, VibrationSensor, ThermalCamera,
    DelaminationRegion, ModalAnalysisResult, GroutingTask,
    Alert, GroutingDiffusion, ReinforcementEffectiveness,
)
from backend.models import get_db
from backend.schemas.models import (
    CaveResponse, WallSurfaceResponse, VibrationSensorResponse,
    ThermalCameraResponse, DelaminationRegionResponse,
    GroutingTaskResponse, AlertResponse, SystemStatusResponse,
)

router = APIRouter(prefix="/inventory", tags=["基础数据管理"])


@router.get("/caves", response_model=List[CaveResponse])
async def list_caves(db: AsyncSession = Depends(get_db)):
    stmt = select(Cave).order_by(Cave.cave_id)
    result = await db.execute(stmt)
    return result.scalars().all()


@router.get("/caves/{cave_id}", response_model=CaveResponse)
async def get_cave(cave_id: str, db: AsyncSession = Depends(get_db)):
    stmt = select(Cave).where(Cave.cave_id == cave_id)
    result = await db.execute(stmt)
    cave = result.scalar_one_or_none()
    if not cave:
        raise HTTPException(status_code=404, detail="洞窟不存在")
    return cave


@router.get("/caves/{cave_id}/surfaces", response_model=List[WallSurfaceResponse])
async def list_cave_surfaces(cave_id: str, db: AsyncSession = Depends(get_db)):
    stmt = select(WallSurface).where(WallSurface.cave_id == cave_id)
    result = await db.execute(stmt)
    return result.scalars().all()


@router.get("/surfaces", response_model=List[WallSurfaceResponse])
async def list_surfaces(
    cave_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(WallSurface)
    if cave_id:
        stmt = stmt.where(WallSurface.cave_id == cave_id)
    stmt = stmt.order_by(WallSurface.surface_id)
    result = await db.execute(stmt)
    return result.scalars().all()


@router.get("/vibration-sensors", response_model=List[VibrationSensorResponse])
async def list_vibration_sensors(
    cave_id: Optional[str] = None,
    surface_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(VibrationSensor)
    if cave_id:
        stmt = stmt.where(VibrationSensor.cave_id == cave_id)
    if surface_id:
        stmt = stmt.where(VibrationSensor.surface_id == surface_id)
    stmt = stmt.order_by(VibrationSensor.sensor_id)
    result = await db.execute(stmt)
    return result.scalars().all()


@router.get("/thermal-cameras", response_model=List[ThermalCameraResponse])
async def list_thermal_cameras(
    cave_id: Optional[str] = None,
    surface_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(ThermalCamera)
    if cave_id:
        stmt = stmt.where(ThermalCamera.cave_id == cave_id)
    if surface_id:
        stmt = stmt.where(ThermalCamera.surface_id == surface_id)
    stmt = stmt.order_by(ThermalCamera.camera_id)
    result = await db.execute(stmt)
    return result.scalars().all()


@router.get("/grouting-tasks", response_model=List[GroutingTaskResponse])
async def list_grouting_tasks(
    cave_id: Optional[str] = None,
    surface_id: Optional[str] = None,
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(GroutingTask)
    if cave_id:
        stmt = stmt.where(GroutingTask.cave_id == cave_id)
    if surface_id:
        stmt = stmt.where(GroutingTask.surface_id == surface_id)
    if status:
        stmt = stmt.where(GroutingTask.status == status)
    stmt = stmt.order_by(desc(GroutingTask.start_time))
    result = await db.execute(stmt)
    return result.scalars().all()


@router.get("/grouting-tasks/{task_id}", response_model=GroutingTaskResponse)
async def get_grouting_task(task_id: str, db: AsyncSession = Depends(get_db)):
    stmt = select(GroutingTask).where(GroutingTask.task_id == task_id)
    result = await db.execute(stmt)
    task = result.scalar_one_or_none()
    if not task:
        raise HTTPException(status_code=404, detail="灌浆任务不存在")
    return task


@router.get("/alerts", response_model=List[AlertResponse])
async def list_alerts(
    cave_id: Optional[str] = None,
    surface_id: Optional[str] = None,
    severity: Optional[str] = None,
    status: Optional[str] = "active",
    days: int = Query(7, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Alert)
    if cave_id:
        stmt = stmt.where(Alert.cave_id == cave_id)
    if surface_id:
        stmt = stmt.where(Alert.surface_id == surface_id)
    if severity:
        stmt = stmt.where(Alert.severity == severity)
    if status:
        stmt = stmt.where(Alert.status == status)
    stmt = stmt.where(Alert.created_at >= datetime.utcnow() - timedelta(days=days))
    stmt = stmt.order_by(desc(Alert.created_at))
    result = await db.execute(stmt)
    return result.scalars().all()


@router.get("/status", response_model=SystemStatusResponse)
async def system_status(db: AsyncSession = Depends(get_db)):
    v_stmt = select(func.count(VibrationSensor.sensor_id)).where(VibrationSensor.status == "active")
    v_count = (await db.execute(v_stmt)).scalar() or 0

    t_stmt = select(func.count(ThermalCamera.camera_id)).where(ThermalCamera.status == "active")
    t_count = (await db.execute(t_stmt)).scalar() or 0

    a_stmt = select(func.count(Alert.alert_id)).where(Alert.status == "active")
    a_count = (await db.execute(a_stmt)).scalar() or 0

    g_stmt = select(func.count(GroutingTask.task_id)).where(GroutingTask.status == "in_progress")
    g_count = (await db.execute(g_stmt)).scalar() or 0

    last_stmt = select(func.max(ModalAnalysisResult.time))
    last_time = (await db.execute(last_stmt)).scalar()

    return {
        "status": "ok",
        "vibration_sensors": v_count,
        "thermal_cameras": t_count,
        "active_alerts": a_count,
        "active_grouting_tasks": g_count,
        "last_processing_time": last_time.isoformat() if last_time else None,
    }
