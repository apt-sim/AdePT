# SPDX-FileCopyrightText: 2026 CERN
# SPDX-License-Identifier: Apache-2.0

SPHINXOPTS ?=
SPHINXBUILD ?= sphinx-build
SOURCEDIR = .
BUILDDIR = _build
DOXYGEN ?= doxygen
PYTHON ?= python3

.PHONY: help clean html runtime-parameters

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	@rm -rf "$(BUILDDIR)"

runtime-parameters:
	@$(PYTHON) scripts/generate_runtime_parameters.py

html:
	@$(PYTHON) scripts/generate_runtime_parameters.py
	@mkdir -p "$(BUILDDIR)/doxygen"
	@$(DOXYGEN) Doxyfile
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
