# 发布流程（RELEASING）

本项目是「本地客户端 + 开源库」形态，发布 = 打一个可安装的 wheel + 一个 git tag + 一份变更日志。
不用发容器镜像也能用；镜像可选。

## 0 前置

- 本机有 Python 3.10+。
- 工作树干净（`git status` 无改动）。

## 1 版本号（三处必须一致）

| 位置 | 字段 |
|---|---|
| `pyproject.toml` | `[project] version` |
| `rosmontis/__init__.py` | `__version__` |
| `README.md` | 如出现版本号处 |

`api_version`（对外接口版本）只有**破坏性改动**时才升；它和包版本不是一回事。

## 2 跑门禁

```bash
set PYTHONIOENCODING=utf-8
python -m unittest discover -s tests -p "test_*.py"      # 必须全绿
python scripts/export_openapi.py                          # 契约快照
git diff --exit-code openapi.json                        # 有 diff 说明接口变了，要么提交要么改回
python bench/run_eval.py                                  # 质量数字
```

## 3 更新变更日志

在 `CHANGELOG.md` 顶部加一节：`## [x.y.z] - YYYY-MM-DD`，按 Added / Changed / Fixed / Removed 分类。
**写“用户能感知的行为变化”，不要写提交日志式的流水账。**

## 4 打包与干净环境验证

```bash
rm -rf dist build *.egg-info
python -m pip wheel . -w dist --no-deps
# 检查产物：必须含 rosmontis/schema.sql，不许含 web/ tests/ rosmontis.db
python -m zipfile -l dist/rosmontis_memory-*.whl

python -m venv %TEMP%\rosmontis_verify
%TEMP%\rosmontis_verify\Scripts\pip install dist\rosmontis_memory-*.whl[serve]
%TEMP%\rosmontis_verify\Scripts\python -c "import rosmontis; print(rosmontis.__version__)"
%TEMP%\rosmontis_verify\Scripts\rosmontis --version
%TEMP%\rosmontis_verify\Scripts\rosmontis serve --port 8878     # 另开窗口 curl /healthz /readyz
```

## 5 打 tag 与发布说明

```bash
git tag -a v1.0.0 -m "v1.0.0"
git push origin master --tags
```

GitHub Release 正文直接取 `CHANGELOG.md` 对应小节；附 `dist/*.whl` 与 `openapi.json`。

## 6 发布后

- 在干净机器/容器里按 README 走一遍「10 分钟接入」。
- 记录发布验证结论到 `bench/report.md` 与 `CHANGELOG.md`。

