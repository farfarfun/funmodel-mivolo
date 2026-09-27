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

### 变更

- 无。

### 废弃

- 无。
