from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any

from backend.models import get_db
from backend.schemas.models import (
    DiffusionSimulateRequest, DiffusionResponse, EffectivenessAssessmentResponse,
)
from backend.services.data_processing import GroutingAnalysisService
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/grouting", tags=["灌浆分析与加固评估"])

grouting_service = GroutingAnalysisService()


@router.post("/simulate-diffusion", response_model=List[DiffusionResponse])
async def simulate_grouting_diffusion(
    req: DiffusionSimulateRequest,
    db: AsyncSession = Depends(get_db),
):
    try:
        results = await grouting_service.simulate_diffusion(
            db, req.task_id, req.elapsed_seconds
        )
        return [DiffusionResponse(**r) for r in results]
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"灌浆扩散模拟失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/assess-effectiveness/{task_id}", response_model=EffectivenessAssessmentResponse)
async def assess_reinforcement(
    task_id: str,
    db: AsyncSession = Depends(get_db),
):
    try:
        result = await grouting_service.assess_effectiveness(db, task_id)
        return EffectivenessAssessmentResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"加固效果评估失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/effectiveness-history/{surface_id}")
async def get_effectiveness_history(
    surface_id: str,
    db: AsyncSession = Depends(get_db),
):
    from backend.models.database import ReinforcementEffectiveness
    from sqlalchemy import select, desc

    stmt = (
        select(ReinforcementEffectiveness)
        .where(ReinforcementEffectiveness.surface_id == surface_id)
        .order_by(desc(ReinforcementEffectiveness.time))
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()

    return [
        {
            "time": r.time.isoformat() if r.time else None,
            "task_id": r.task_id,
            "frequency_recovery_pct": float(r.frequency_recovery_pct) if r.frequency_recovery_pct else None,
            "delamination_area_reduction_pct": float(r.delamination_area_reduction_pct) if r.delamination_area_reduction_pct else None,
            "bonding_strength_mpa": float(r.bonding_strength_mpa) if r.bonding_strength_mpa else None,
            "overall_score": float(r.overall_score) if r.overall_score else None,
            "assessment_notes": r.assessment_notes,
        }
        for r in rows
    ]
