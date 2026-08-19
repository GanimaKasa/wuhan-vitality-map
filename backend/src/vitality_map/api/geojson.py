from fastapi import APIRouter
from fastapi.responses import JSONResponse

from vitality_map.core.data import GEOJSON, KNOWN_DISTRICTS, STUDY_AREA_BOUNDARY

router = APIRouter(tags=["geojson"])


@router.get("/api/geojson")
def get_geojson():
    return JSONResponse(content=GEOJSON)


@router.get("/api/study_area_boundary")
def get_study_area_boundary():
    """三环内研究区域的整体边界线（单个多边形轮廓，不是逐格网边框），
    数据来自原始数据/shp文件/武汉三环面.shp转出的GeoJSON。"""
    return JSONResponse(content=STUDY_AREA_BOUNDARY)


@router.get("/api/districts")
def get_districts():
    return {"districts": KNOWN_DISTRICTS}
