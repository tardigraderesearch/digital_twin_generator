# Virtual bench (sim/): an end-to-end simulated hardware test. Private, like the rest of DTG.
# The sim environment is separate from the app's .venv: it also needs robo-prober (v0.2,
# for the real host API and firmware core) and aruco-tag-maker (tag bitmaps + markers.json).
ROBO_PROBER ?= $(HOME)/Documents/github-portfolio-review/staging/robo-prober
ARUCO_TAG_MAKER ?= $(HOME)/Documents/github-portfolio-review/staging/aruco-tag-maker
SIM_PY := sim/.venv/bin/python
SEED ?= 0
N ?= 50
WORKERS ?= 16

.PHONY: sim sim-env sim-run sim-video sim-mc sim-test sim-gui

sim: sim-env sim-test sim-run          ## quick check: harness tests + one robot end to end

sim-env: $(SIM_PY)
$(SIM_PY):
	python3 -m venv sim/.venv
	$(SIM_PY) -m pip install -q -e ".[test]" -e "$(ROBO_PROBER)" manifold3d
	$(SIM_PY) -m pip install -q --no-deps -e "$(ARUCO_TAG_MAKER)"

sim-test: sim-env
	$(SIM_PY) -m pytest -q sim/tests

sim-run: sim-env
	$(SIM_PY) -m sim run --seed $(SEED)

sim-video: sim-env
	$(SIM_PY) -m sim run --seed $(SEED) --light medium --video --out runs/sim-video

sim-mc: sim-env                        ## N robots in parallel, then sim/REPORT.md
	$(SIM_PY) -m sim montecarlo --n $(N) --workers $(WORKERS) --out runs/sim-mc --report

sim-gui: sim-env                       ## needs an X display (window stays withdrawn)
	$(SIM_PY) -m sim gui-smoke --seed $(SEED)
