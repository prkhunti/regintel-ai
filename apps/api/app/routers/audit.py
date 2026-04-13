"""
Audit log endpoints.

GET /audit/events        — paginated list with optional filters
GET /audit/events/{id}   — single event by ID
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models import AuditEvent
from packages.schemas.audit import AuditEventRead, AuditEventType

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("/events", response_model=list[AuditEventRead])
async def list_audit_events(
    event_type: Annotated[AuditEventType | None, Query(description="Filter by event type")] = None,
    resource_type: Annotated[str | None, Query(description="Filter by resource type (e.g. 'response', 'document')")] = None,
    resource_id: Annotated[uuid.UUID | None, Query(description="Filter by specific resource UUID")] = None,
    from_dt: Annotated[datetime | None, Query(alias="from", description="ISO-8601 start timestamp (inclusive)")] = None,
    to_dt: Annotated[datetime | None, Query(alias="to", description="ISO-8601 end timestamp (inclusive)")] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    db: AsyncSession = Depends(get_db),
):
    """
    List audit events newest-first with optional filters.

    All query parameters are combinable. Results are always ordered by
    created_at DESC so the most recent events appear first.
    """
    stmt = select(AuditEvent).order_by(AuditEvent.created_at.desc())

    if event_type is not None:
        stmt = stmt.where(AuditEvent.event_type == str(event_type))
    if resource_type is not None:
        stmt = stmt.where(AuditEvent.resource_type == resource_type)
    if resource_id is not None:
        stmt = stmt.where(AuditEvent.resource_id == resource_id)
    if from_dt is not None:
        stmt = stmt.where(AuditEvent.created_at >= from_dt)
    if to_dt is not None:
        stmt = stmt.where(AuditEvent.created_at <= to_dt)

    stmt = stmt.offset(offset).limit(limit)
    result = await db.execute(stmt)
    rows = result.scalars().all()
    return [AuditEventRead.model_validate(r) for r in rows]


@router.get("/events/count")
async def count_audit_events(
    event_type: Annotated[AuditEventType | None, Query()] = None,
    resource_type: Annotated[str | None, Query()] = None,
    from_dt: Annotated[datetime | None, Query(alias="from")] = None,
    to_dt: Annotated[datetime | None, Query(alias="to")] = None,
    db: AsyncSession = Depends(get_db),
) -> dict[str, int]:
    """Return total count matching the same filters as /events."""
    stmt = select(func.count()).select_from(AuditEvent)

    if event_type is not None:
        stmt = stmt.where(AuditEvent.event_type == str(event_type))
    if resource_type is not None:
        stmt = stmt.where(AuditEvent.resource_type == resource_type)
    if from_dt is not None:
        stmt = stmt.where(AuditEvent.created_at >= from_dt)
    if to_dt is not None:
        stmt = stmt.where(AuditEvent.created_at <= to_dt)

    total = (await db.execute(stmt)).scalar_one()
    return {"count": total}


@router.get("/events/{event_id}", response_model=AuditEventRead)
async def get_audit_event(
    event_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Fetch a single audit event by its UUID."""
    row = await db.get(AuditEvent, event_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Audit event not found")
    return AuditEventRead.model_validate(row)
