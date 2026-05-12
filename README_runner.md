# 饮料瓶残留液体识别项目一键执行脚本

## 推荐放置结构

```text
waterornot/
├── run_all.bat
├── run_all.sh
└── scripts/
    └── 12_run_all.py
```

## 常用命令

完整运行：

```bash
python scripts/12_run_all.py --mode all
```

只做数据准备：

```bash
python scripts/12_run_all.py --mode prepare
```

只训练：

```bash
python scripts/12_run_all.py --mode train
```

只评价：

```bash
python scripts/12_run_all.py --mode eval
```

只端到端预测：

```bash
python scripts/12_run_all.py --mode predict
```

只生成可视化图表：

```bash
python scripts/12_run_all.py --mode visualize
```

只预览流程，不执行：

```bash
python scripts/12_run_all.py --mode all --dry-run
```
