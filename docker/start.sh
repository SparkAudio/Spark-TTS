#!/bin/bash

# 下载模型
python download_models.py

# 构建并启动Docker容器
docker compose up -d
