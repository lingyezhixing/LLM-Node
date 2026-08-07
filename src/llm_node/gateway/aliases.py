"""Alias → primary_name 解析,OpenAI 兼容代理与管理 API 共用。"""

from __future__ import annotations

from fastapi import HTTPException

from llm_node import config


def resolve_alias_checked(cfg: config.AppConfig, alias: str | None) -> str:
    """alias → primary_name。缺失 → 400;未知别名 → 404。"""
    if not alias:
        raise HTTPException(400, "请求体(JSON)中缺少 'model' 字段")
    try:
        return config.resolve_alias(cfg, alias)
    except KeyError:
        raise HTTPException(404, f"模型别名 '{alias}' 未在配置中找到")
