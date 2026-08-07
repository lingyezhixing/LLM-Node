#!/usr/bin/env python3
"""
LLM-Node 计算节点主程序(无状态后端,对齐 LLM-Manager v3 架构)。
启动方式:python main.py 或 python -m llm_node(后者需已安装)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from llm_node.app import run

if __name__ == "__main__":
    run()
