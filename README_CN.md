<div align="center">
    <h1>
    Spark-TTS
    </h1>
    <p>
    <b><em>Spark-TTS: 一种基于单流解耦语音标记的高效LLM文本转语音模型</em></b>的官方PyTorch推理代码
    </p>
    <p>
    <img src="src/logo/SparkTTS.jpg" alt="Spark-TTS Logo" style="width: 200px; height: 200px;">
    </p>
        <p>
        <img src="src/logo/HKUST.jpg" alt="Institution 1" style="width: 200px; height: 60px;">
        <img src="src/logo/mobvoi.jpg" alt="Institution 2" style="width: 200px; height: 60px;">
        <img src="src/logo/SJU.jpg" alt="Institution 3" style="width: 200px; height: 60px;">
    </p>
    <p>
        <img src="src/logo/NTU.jpg" alt="Institution 4" style="width: 200px; height: 60px;">
        <img src="src/logo/NPU.jpg" alt="Institution 5" style="width: 200px; height: 60px;">
        <img src="src/logo/SparkAudio2.jpg" alt="Institution 6" style="width: 200px; height: 60px;">
    </p>
    <p>
    </p>
    <a href="https://arxiv.org/pdf/2503.01710"><img src="https://img.shields.io/badge/Paper-ArXiv-red" alt="paper"></a>
    <a href="https://sparkaudio.github.io/spark-tts/"><img src="https://img.shields.io/badge/Demo-Page-lightgrey" alt="version"></a>
    <a href="https://huggingface.co/SparkAudio/Spark-TTS-0.5B"><img src="https://img.shields.io/badge/Hugging%20Face-Model%20Page-yellow" alt="Hugging Face"></a>
    <a href="https://github.com/SparkAudio/Spark-TTS"><img src="https://img.shields.io/badge/Platform-linux-lightgrey" alt="version"></a>
    <a href="https://github.com/SparkAudio/Spark-TTS"><img src="https://img.shields.io/badge/Python-3.12+-orange" alt="version"></a>
    <a href="https://github.com/SparkAudio/Spark-TTS"><img src="https://img.shields.io/badge/PyTorch-2.5+-brightgreen" alt="python"></a>
    <a href="https://github.com/SparkAudio/Spark-TTS"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="mit"></a>
</div>


## Spark-TTS 🔥

### 概述

Spark-TTS是一个先进的文本转语音系统，利用大型语言模型(LLM)的强大功能实现高度准确且自然的语音合成。它被设计为高效、灵活且功能强大，适用于研究和生产环境。

### 主要特点

- **简洁高效**：完全基于Qwen2.5构建，无需额外的生成模型如flow matching。不同于依赖独立模型生成声学特征，它直接从LLM预测的代码重建音频。这种方法简化了流程，提高了效率并降低了复杂性。
- **高质量声音克隆**：支持零样本声音克隆，即使没有特定声音的训练数据也能复制说话者的声音。这在跨语言和代码切换场景中尤为理想，允许模型在不同语言和声音间无缝转换，而无需为每种情况单独训练。
- **双语支持**：同时支持中文和英文，能够在跨语言和代码切换场景中进行零样本声音克隆，使模型能够以高自然度和准确性合成多种语言的语音。
- **可控语音生成**：支持通过调整性别、音高和语速等参数创建虚拟说话者。

---

<table align="center">
  <tr>
    <td align="center"><b>声音克隆推理概述</b><br><img src="src/figures/infer_voice_cloning.png" width="80%" /></td>
  </tr>
  <tr>
    <td align="center"><b>可控生成推理概述</b><br><img src="src/figures/infer_control.png" width="80%" /></td>
  </tr>
</table>


## 🚀 新闻

- **[2025-03-04]** 我们的项目论文已发布！您可以在这里阅读：[Spark-TTS](https://arxiv.org/pdf/2503.01710)。

- **[2025-03-12]** 现已支持Nvidia Triton推理服务。详情请参阅下面的运行时部分。


## 安装

### 标准安装

**克隆并安装**

  以下是在Linux上安装的说明。如果您使用Windows，请参考[Windows安装指南](https://github.com/SparkAudio/Spark-TTS/issues/5)。  
*(感谢 [@AcTePuKc](https://github.com/AcTePuKc) 提供详细的Windows安装指南！)*


- 克隆仓库
``` sh
git clone https://github.com/SparkAudio/Spark-TTS.git
cd Spark-TTS
```

- 安装Conda：请参见 https://docs.conda.io/en/latest/miniconda.html
- 创建Conda环境：

``` sh
conda create -n sparktts -y python=3.12
conda activate sparktts
pip install -r requirements.txt
# 如果您在中国大陆，可以按如下方式设置镜像：
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
```

**模型下载**

通过Python下载：
```python
from huggingface_hub import snapshot_download

snapshot_download("SparkAudio/Spark-TTS-0.5B", local_dir="pretrained_models/Spark-TTS-0.5B")
```

通过git clone下载：
```sh
mkdir -p pretrained_models

# 确保已安装git-lfs（https://git-lfs.com）
git lfs install

git clone https://huggingface.co/SparkAudio/Spark-TTS-0.5B pretrained_models/Spark-TTS-0.5B
```

### Docker安装

Spark-TTS提供Docker支持，便于设置和部署。这对于生产环境或避免依赖冲突特别有帮助。

**先决条件：**
- 系统已安装Docker和Docker Compose
- 对于GPU支持：已配置NVIDIA Docker运行时

**使用Docker运行的步骤：**

1. 克隆仓库：
```sh
git clone https://github.com/SparkAudio/Spark-TTS.git
cd Spark-TTS/docker
```

2. 运行启动脚本，该脚本会自动下载模型并构建容器：
```sh
chmod +x start.sh
./start.sh
```

3. 在浏览器中访问Web UI，地址为 http://localhost:12370。

**常用Docker命令：**
```sh
# 在后台启动容器
docker compose up -d

# 查看容器日志
docker compose logs -f

# 停止容器
docker compose down

# 重新构建并重启容器
docker compose up -d --build
```

**注意：**

- 模型文件存储在`pretrained_models`目录中
- 生成的音频文件保存在`example/results`中
- 如果不需要GPU支持，请在`./docker/docker-compose.yml`中注释掉GPU相关配置
- 更多说明可在`./docker/Docker_README.md`中找到


## 基本用法

您可以通过以下命令简单地运行演示：
``` sh
cd example
bash infer.sh
```

或者，您可以直接在命令行中执行以下命令进行推理：

``` sh
python -m cli.inference \
    --text "要合成的文本。" \
    --device 0 \
    --save_dir "保存音频的路径" \
    --model_dir pretrained_models/Spark-TTS-0.5B \
    --prompt_text "提示音频的文本" \
    --prompt_speech_path "提示音频的路径"
```

**Web UI使用**

您可以通过运行`python webui.py --device 0`启动UI界面，该界面允许您执行声音克隆和声音创建。声音克隆支持上传参考音频或直接录制音频。

如果使用Docker，只需在启动容器后在浏览器中访问 http://localhost:12370。


| **声音克隆** | **声音创建** |
|:-------------------:|:-------------------:|
| ![Image 1](src/figures/gradio_TTS.png) | ![Image 2](src/figures/gradio_control.png) |


**可选方法**

有关其他CLI和Web UI方法，包括替代实现和扩展功能，您可以参考：

- [AcTePuKc的CLI和UI](https://github.com/SparkAudio/Spark-TTS/issues/10)


## 运行时

**Nvidia Triton推理服务**

我们现在提供使用Nvidia Triton和TensorRT-LLM部署Spark-TTS的参考。下表展示了在单个L20 GPU上使用26对不同提示音频/目标文本对（总计169秒音频）的基准测试结果：

| 模型 | 说明 | 并发数 | 平均延迟 | RTF | 
|-------|-----------|-----------------------|---------|--|
| Spark-TTS-0.5B | [代码提交](https://github.com/SparkAudio/Spark-TTS/tree/4d769ff782a868524f29e0be851ca64f8b22ebf1/runtime/triton_trtllm) | 1 | 876.24 ms | 0.1362|
| Spark-TTS-0.5B | [代码提交](https://github.com/SparkAudio/Spark-TTS/tree/4d769ff782a868524f29e0be851ca64f8b22ebf1/runtime/triton_trtllm) | 2 | 920.97 ms | 0.0737|
| Spark-TTS-0.5B | [代码提交](https://github.com/SparkAudio/Spark-TTS/tree/4d769ff782a868524f29e0be851ca64f8b22ebf1/runtime/triton_trtllm) | 4 | 1611.51 ms | 0.0704|


更多信息请参见[runtime/triton_trtllm/README.md](runtime/triton_trtllm/README.md)。


## **演示**

以下是使用Spark-TTS零样本声音克隆生成的一些演示。更多演示，请访问我们的[演示页面](https://sparkaudio.github.io/spark-tts/)。

---

<table>
<tr>
<td align="center">

**唐纳德·特朗普**
</td>
<td align="center">
    
**钟离（原神）**
</td>
</tr>

<tr>
<td align="center">

[Donald Trump](https://github.com/user-attachments/assets/fb225780-d9fe-44b2-9b2e-54390cb3d8fd)

</td>
<td align="center">
    
[Zhongli](https://github.com/user-attachments/assets/80eeb9c7-0443-4758-a1ce-55ac59e64bd6)

</td>
</tr>
</table>

---

<table>

<tr>
<td align="center">
    
**陈鲁豫**
</td>
<td align="center">
    
**杨澜**
</td>
</tr>

<tr>
<td align="center">
    
[陈鲁豫Chen_Luyu.webm](https://github.com/user-attachments/assets/5c6585ae-830d-47b1-992d-ee3691f48cf4)
</td>
<td align="center">
    
[Yang_Lan.webm](https://github.com/user-attachments/assets/2fb3d00c-abc3-410e-932f-46ba204fb1d7)
</td>
</tr>
</table>

---


<table>
<tr>
<td align="center">

**余承东**
</td>
<td align="center">
    
**马云**
</td>
</tr>

<tr>
<td align="center">

[Yu_Chengdong.webm](https://github.com/user-attachments/assets/78feca02-84bb-4d3a-a770-0cfd02f1a8da)

</td>
<td align="center">
    
[Ma_Yun.webm](https://github.com/user-attachments/assets/2d54e2eb-cec4-4c2f-8c84-8fe587da321b)

</td>
</tr>
</table>

---


<table>
<tr>
<td align="center">

**刘德华**
</td>
<td align="center">

**徐志胜**
</td>
</tr>

<tr>
<td align="center">

[Liu_Dehua.webm](https://github.com/user-attachments/assets/195b5e97-1fee-4955-b954-6d10fa04f1d7)

</td>
<td align="center">
    
[Xu_Zhisheng.webm](https://github.com/user-attachments/assets/dd812af9-76bd-4e26-9988-9cdb9ccbb87b)

</td>
</tr>
</table>


---

<table>
<tr>
<td align="center">

**哪吒**
</td>
<td align="center">
    
**李靖**
</td>
</tr>

<tr>
<td align="center">

[Ne_Zha.webm](https://github.com/user-attachments/assets/8c608037-a17a-46d4-8588-4db34b49ed1d)
</td>
<td align="center">

[Li_Jing.webm](https://github.com/user-attachments/assets/aa8ba091-097c-4156-b4e3-6445da5ea101)

</td>
</tr>
</table>


## 待办事项

- [x] 发布Spark-TTS论文。
- [ ] 发布训练代码。
- [ ] 发布训练数据集VoxBox。


## 引用

```
@misc{wang2025sparktts,
      title={Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens}, 
      author={Xinsheng Wang and Mingqi Jiang and Ziyang Ma and Ziyu Zhang and Songxiang Liu and Linqin Li and Zheng Liang and Qixi Zheng and Rui Wang and Xiaoqin Feng and Weizhen Bian and Zhen Ye and Sitong Cheng and Ruibin Yuan and Zhixian Zhao and Xinfa Zhu and Jiahao Pan and Liumeng Xue and Pengcheng Zhu and Yunlin Chen and Zhifei Li and Xie Chen and Lei Xie and Yike Guo and Wei Xue},
      year={2025},
      eprint={2503.01710},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2503.01710}, 
}
```


## ⚠️ 使用免责声明

本项目提供一个零样本声音克隆TTS模型，旨在用于学术研究、教育目的和合法应用，如个性化语音合成、辅助技术和语言研究。

请注意：

- 请勿将此模型用于未经授权的声音克隆、冒充、欺诈、诈骗、深度伪造或任何非法活动。

- 使用此模型时请确保遵守当地法律法规，并坚持道德标准。

- 开发者对此模型的任何滥用不承担责任。

我们主张负责任地开发和使用AI，并鼓励社区在AI研究和应用中坚持安全和道德原则。如果您对道德或滥用有任何疑虑，请联系我们。