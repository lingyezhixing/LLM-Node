@echo off
chcp 65001 >nul
title LLM-Manager

:: 设置环境变量以确保 Python 使用 UTF-8 编码
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

:: 激活conda环境并运行（日志管理由Python程序自动处理）
call conda activate LLM-Manager
python main.py