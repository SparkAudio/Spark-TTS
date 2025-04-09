FROM python:3.12-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    git-lfs \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 安装git-lfs并初始化
RUN git lfs install

# 复制项目文件
COPY . /app/

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 创建模型目录
RUN mkdir -p pretrained_models

# 设置容器启动时运行的命令
CMD ["python", "webui.py", "--device", "0", "--server_name", "0.0.0.0"]

# 暴露端口
EXPOSE 7860