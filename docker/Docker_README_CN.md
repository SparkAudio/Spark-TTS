# Spark-TTS Docker 使用指南

本目录包含了在Docker中运行Spark-TTS项目的所有必要文件。

## 目录结构

```
docker/
├── Dockerfile           # Docker镜像构建文件
├── docker-compose.yml   # Docker Compose配置文件
├── .dockerignore        # Docker构建忽略文件
├── download_models.py   # 下载预训练模型的脚本
├── start.sh             # 启动Docker容器的脚本
└── README.md            # 本文档
```

## 快速开始

1. 确保您已安装Docker和Docker Compose
2. 在docker目录下运行启动脚本：
   ```bash
   sudo chmod +x start.sh
   ./start.sh
   ```
   
   这个脚本会：
   - 自动下载模型，如果没有则下载
   - 构建并启动Docker容器（后台模式）

3. 打开浏览器，访问 `http://localhost:12370` 使用WebUI。

## 常用命令

```bash
# 启动容器（后台运行）
docker compose up -d

# 查看容器日志
docker compose logs -f

# 停止容器
docker compose down

# 重新构建并启动容器
docker compose up -d --build
```

## 注意事项

1. 模型文件会下载到项目根目录的`pretrained_models`目录
2. 生成的音频文件保存在项目根目录的`example/results`目录
3. 如果不需要GPU支持，请在`docker-compose.yml`中注释掉GPU相关配置

## 故障排除

如果遇到问题，请检查：

1. Docker和Docker Compose是否正确安装
2. 如果使用GPU，NVIDIA Docker Runtime是否正确配置
3. 网络连接是否正常（下载模型需要连接互联网）
4. 端口12370是否已被其他应用占用
