# 参与贡献

## 开发环境

```bash
git clone <repo>
cd rosmontis-memory
python -m venv .venv
.venv\Scripts\activate            # Windows；POSIX 用 source .venv/bin/activate
pip install -e ".[dev]"
set PYTHONIOENCODING=utf-8        # Windows 控制台
```

## 提交前必须通过

```bash
python -m unittest discover -s tests -p "test_*.py"     # 全绿，判定以 “Ran N tests … OK” 为准
python scripts/export_openapi.py                        # 接口有改动时重导契快照
git diff --exit-code openapi.json                        # 快照与代码一致
python examples/hooks_demo.py                            # 端到端示例可跑
```

## 约定

- **接口改动**：`/v1/*` 与公共 API 属冻结面，破坏性改动必须新增强版本路径并升 `api_version`；`openapi.json` 与代码同步提交。
- **schema 改动**：走版本化迁移，单事务、幂等、失败回滚；不得直接改历史列语义。
- **测试**：新功能配新测试；不要靠删测试或放宽断言过关。
- **隐私**：不要提交真实对话、真实姓名、密钥或本机路径；示例与基准集使用中性化数据。
- **文档**：陈述事实与契约；历史变更交给 git，不写“本次更新”式流水账。
- **代码注释**：写“代码做什么”，不写“为什么被要求改”。

## 提交信息

一句话说清「改了什么、影响哪一层」。涉及行为变化时写清用户可感知的部分。

## 报告问题

安全漏洞走 `SECURITY.md`；其他问题请附最小复现与版本号（`rosmontis --version`）。
