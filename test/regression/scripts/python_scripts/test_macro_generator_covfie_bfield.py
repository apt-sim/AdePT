#! /usr/bin/env python3

# SPDX-FileCopyrightText: 2026 CERN
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import tempfile
from pathlib import Path


def run_generator(output, *extra_args):
    script_dir = Path(__file__).resolve().parent
    scripts_dir = script_dir.parent
    subprocess.run(
        [
            sys.executable,
            str(script_dir / "macro_generator.py"),
            "--template",
            str(scripts_dir / "example_template.mac"),
            "--output",
            str(output),
            "--gdml_name",
            "/tmp/testEm3.gdml",
            "--num_threads",
            "1",
            "--num_events",
            "1",
            "--track_in_all_regions",
            "True",
            "--gun_type",
            "setDefault",
            *extra_args,
        ],
        check=True,
    )


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        covfie_macro = tmp_dir / "covfie.mac"
        run_generator(
            covfie_macro,
            "--detector_field",
            "0 0 1",
            "--covfie_bfield_file",
            "/tmp/constant_1T.cvf",
        )
        covfie_content = covfie_macro.read_text()
        assert "/detector/setCovfieBfieldFile /tmp/constant_1T.cvf" in covfie_content
        assert "/adept/setCovfieBfieldFile /tmp/constant_1T.cvf" in covfie_content
        assert "/detector/setField" not in covfie_content

        constant_macro = tmp_dir / "constant.mac"
        run_generator(constant_macro, "--detector_field", "0 0 1")
        constant_content = constant_macro.read_text()
        assert "/detector/setField 0 0 1 tesla" in constant_content
        assert "setCovfieBfieldFile" not in constant_content


if __name__ == "__main__":
    main()
