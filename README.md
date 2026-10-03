# funmodel-mivolo

基于 [WildChlamydia/MiVOLO](https://github.com/WildChlamydia/MiVOLO) 封装的人脸/人体年龄与性别识别模型，作为 `funmodel` 命名空间下的 `funmodel.mivolo` 子包发布，提供开箱即用的预测器（自动下载权重、YOLO 检测 + MiVOLO 分类）。

## 安装

```bash
uv add funmodel-mivolo
# 或
pip install funmodel-mivolo
```

## 最小示例

```python
from funmodel.mivolo import MivoloPredictor
from funmodel.utils.image.transform import url_to_cvimg

predictor = MivoloPredictor(device="cpu")
image = url_to_cvimg(
    "https://raw.githubusercontent.com/WildChlamydia/MiVOLO/"
    "main/images/MiVOLO.jpg"
)
results, _ = predictor.predict(image)
print(results)  # [{"age": ..., "gender": ..., "gender_score": ..., ...}, ...]
```

首次调用会自动从组织 OSS 下载 `mivolo_imbd.pth.tar` 和 `yolov8x_person_face.pt` 两个权重文件到本地缓存目录。

## 第三方代码与许可证

本项目基于 [WildChlamydia/MiVOLO](https://github.com/WildChlamydia/MiVOLO)
封装，并包含从 [timm](https://github.com/huggingface/pytorch-image-models) 改编的实现。
MiVOLO 和 timm 均采用 [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)；
相关源文件保留了原始版权、许可证和修改声明。本项目其余代码采用 [MIT](LICENSE) 协议。

---

## 关于 farfarfun

[farfarfun](https://github.com/farfarfun) 是一个专注于实用工具库的开源组织，
涵盖云存储、数据处理、AI、多媒体与开发工具链等方向。

- 🏠 组织主页：<https://github.com/farfarfun>
- 📦 PyPI：<https://pypi.org/user/niuliangtao/>
- 📧 联系：farfarfun@qq.com

本项目基于 [MIT](LICENSE) 协议开源。
