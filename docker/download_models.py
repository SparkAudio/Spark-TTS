#!/usr/bin/env python
"""
用于下载Spark-TTS预训练模型的脚本
"""
import os
import sys

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from huggingface_hub import snapshot_download

def download_model(model_name="SparkAudio/Spark-TTS-0.5B", local_dir="../pretrained_models/Spark-TTS-0.5B"):
    """下载Spark-TTS预训练模型"""
    print(f"正在下载模型 {model_name} 到 {local_dir}...")
    
    # 确保目录存在
    os.makedirs(os.path.dirname(local_dir), exist_ok=True)
    
    # 使用huggingface_hub下载模型
    snapshot_download(model_name, local_dir=local_dir)
    
    print(f"模型下载完成！")

if __name__ == "__main__":
    download_model()
