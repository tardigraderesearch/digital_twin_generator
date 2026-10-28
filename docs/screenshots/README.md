# Virtual bench screenshots

Captures of the DTG window driving the virtual robot and firmware simulator
through all eight steps.

1. [Scan](1-scan.png)
2. [Tags](2-tags.png)
3. [Wiggle](3-wiggle.png)
4. [Range of motion](4-range-of-motion.png)
5. [Register](5-register.png)
6. [Cut links](6-cut-links.png)
7. [Missing parts](7-missing-parts.png)
8. [Export](8-export.png)

Reproduce from the repository with a working display:

```sh
make sim-env
sim/.venv/bin/python -m sim gui-walk --seed 2 --screenshots docs/screenshots --out runs/gui-walk
```
