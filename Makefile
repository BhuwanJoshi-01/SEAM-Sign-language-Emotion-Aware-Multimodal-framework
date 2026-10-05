.DEFAULT_GOAL := help
SHELL := /bin/bash
PY ?= /home/bhuwan/miniconda3/envs/slr/bin/python
PIP ?= /home/bhuwan/miniconda3/envs/slr/bin/pip
# Licence-gated SMPL-X parameters. Deliberately not vendored: the Max Planck Institute
# gates the download, so the loader takes a path. Override on the command line.
SMPLX ?= /mnt/Volume2/SignLanguagge/NSL Data/Sapien_Pipeline/models/smplx/SMPLX_NEUTRAL.npz

.PHONY: help setup lint fmt typecheck test test-fast data readiness emosign landmarks facegate \
        doctor bench train repro provenance alignment paper clean distclean demo-avatar \
        avatar-check share \
        fetch-how2sign wlasl-index

help: ## List targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Editable install + pre-commit hooks
	$(PIP) install -e ".[dev]"
	$(PY) -m pre_commit install

lint: ## ruff check + format check
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

fmt: ## Apply ruff formatting
	$(PY) -m ruff format src tests
	$(PY) -m ruff check --fix src tests

typecheck: ## mypy
	$(PY) -m mypy src

test: ## Full test suite
	$(PY) -m pytest

test-fast: ## Test suite minus slow and network tests
	$(PY) -m pytest -m "not slow and not network"

doctor: ## Environment report: GPU, VRAM, mediapipe, ffmpeg, disk, network
	$(PY) -m seam.cli doctor

data: ## Fetch/verify every dataset resource
	$(PY) -m seam.cli data fetch --all
	$(PY) -m seam.cli data verify --all

readiness: ## Print the readiness table with blockers and owners  [M0 GATE]
	$(PY) -m seam.cli data verify --all --table

emosign: ## Report the EmoSign label distribution and the ASLLRP join
	$(PY) -m seam.cli data emosign

fetch-how2sign: ## Fetch the 31 How2Sign pose shards (14.1 GB, resumable)  [M5]
	$(PY) scripts/fetch_how2sign.py

wlasl-index: ## Report the WLASL on-disk index, incl. HTML-placeholder substitutions
	PYTHONPATH=src $(PY) -c "from pathlib import Path; from seam.data.wlasl import index_on_disk, is_placeholder; \
		r=Path('/home/bhuwan/Videos/wlasl/videos'); s={}; i=index_on_disk(r,s); \
		print(f'{len(i)} keys, {len(s)} substitutions, {sum(1 for v in i.values() if is_placeholder(v))} unrecoverable'); \
		[print('  ',k) for k,v in sorted(i.items()) if is_placeholder(v)]"

landmarks: ## Extract landmarks + blendshapes for the EmoSign 200
	$(PY) -m seam.cli landmarks extract --dataset emosign

facegate: ## Face-visibility gate over sampled EmoSign clips  [M0 GATE]
	$(PY) -m seam.cli landmarks face-gate --sample 24

bench: ## Latency / VRAM harness on the RTX 3050  [M2]
	$(PY) -m seam.cli bench

bench-seq: ## Sequential perception baseline for the concurrency ablation  [M2]
	$(PY) -m seam.cli bench --sequential

export: ## ONNX export + FP32/INT8 parity; non-zero on a failed gate  [M2]
	$(PY) -m seam.cli export

demo-avatar: ## Video -> SMPLer-X vs landmark avatar comparison, both arms  [M7]
	PYTHONPATH=src $(PY) scripts/make_avatar_demo.py --model "$(SMPLX)" --limit 4

serve: ## Live browser demo; video is processed client-side, never uploaded  [M7]
	$(PY) -m seam.cli serve

serve-check: ## Boot the demo server and assert the HTTP contract  [M7]
	$(PY) scripts/serve_smoke.py

avatar-check: ## Play every avatar clip in the front page's viewer, in headless Chrome  [M7]
	$(PY) scripts/avatar_smoke.py

share: ## Pack a zip a teammate can run on Windows or Linux without training anything  [release]
	PYTHONPATH=src $(PY) scripts/make_share_bundle.py

parity-report: ## Print the last parity verdict  [M2]
	@$(PY) -c "import json,pathlib; d=json.loads(pathlib.Path('artifacts/export/parity.json').read_text()); \
	[print(r['summary']()) for r in d.get('results_summary',[])]; \
	print('providers:', d['execution_providers']['active_on_probe']); \
	print('inputs:', d['parity_input_source'])"

train: ## Train from configs/
	$(PY) -m seam.cli train

repro: ## Regenerate every artifact the paper cites, in dependency order  [M9]
	bash scripts/repro_all.sh

figures: ## Redraw the report figures in docs/figures from the result artifacts
	$(PY) scripts/make_report_figures.py

site-check: ## Run the standalone live page in headless Chrome with a fake camera (VIDEO=path)
	$(PY) scripts/site_smoke.py --video $(VIDEO)

provenance: ## Fail on an untraced number, or a result written by code that has since changed
	$(PY) -m pytest tests/test_provenance.py tests/test_artifact_staleness.py -v

alignment: ## ASLLRP token-to-crop-frame alignment report (M5a gate)
	$(PY) scripts/check_asllrp_alignment.py

paper: ## Build the paper PDF
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

clean: ## Remove caches; keeps artifacts/ and data
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

distclean: ## clean + artifacts/
	$(MAKE) clean
	rm -rf artifacts
