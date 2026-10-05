# Changelog

## [0.0.10]

### 新增

- 补齐 README（简介、安装命令、最小示例）与组织介绍区块。
- 新增 `.gitignore` 规则：`*.db`、`*.rar`、`.run/`、`logs/`、`.idea/`、`.vscode/`。

### 修复

- 依赖补充版本下限，新增 `farlog` 依赖并生成 `uv.lock`。
- `MivoloPredictor` 公开 API 补充类型标注与中文 docstring。
- 模型与数据读取模块中的 `logging`/`print` 诊断输出统一改为 `farlog`。
- 模型权重下载失败时现在会携带下载地址和目标路径抛出明确错误。
- `age_gender_dataset.py`/`reader_age_gender.py` 中过宽的异常捕获改为捕获具体异常类型并保留上下文。
- 修复 `funmodel/mivolo/data/dataset/__init__.py` 及 `example/pre/` 下数据准备/评估脚本中
  残留的 `from mivolo.xxx import yyy` 旧导入路径（实际顶层模块是 `funmodel.mivolo`），
  此前 `build()` 一调用即抛 `ModuleNotFoundError`。
- 为 `PersonAndFaceCrops`/`PersonAndFaceResult`/`Detector`/`MiVOLO` 等公开类及
  `aggregate_votes_winsorized`/`box_iou`/`build` 等公开函数补充中文 docstring 与类型标注。
- 删除未被任何构建流程消费的残留版本号文件 `funmodel/mivolo/version.py`（真实版本号唯一来源是
  `pyproject.toml`）。
- `py.typed` 标记从命名空间共享目录 `funmodel/` 移动到本包实际代码根目录
  `funmodel/mivolo/`，避免与 `funmodel`/`funmodel-dwpose` 等同命名空间插件包产生
  构建产物路径冲突。

### 变更

- 无。

### 废弃

- 无。
