# SPDX-FileCopyrightText: 2026 CERN
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

project = "AdePT"
copyright = "CERN"
extensions = ["myst_parser", "breathe"]
source_suffix = {".md": "markdown"}
exclude_patterns = ["_build"]
html_theme = "sphinx_rtd_theme"

breathe_projects = {
    "AdePT": str((Path(__file__).parent / "_build" / "doxygen" / "xml").resolve()),
}
breathe_default_project = "AdePT"
