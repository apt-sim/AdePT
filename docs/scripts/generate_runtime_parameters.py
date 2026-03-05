#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CERN
# SPDX-License-Identifier: Apache-2.0

"""Generate docs/user/runtime-parameters.md from source-defined UI commands.

Command metadata source:
  - src/AdePTConfigurationMessenger.cc
Example invocations source:
  - test/regression/scripts/test_ui_commands_template.mac
"""

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path


TYPE_MAP = {
    "G4UIcmdWithABool": "bool",
    "G4UIcmdWithAnInteger": "integer",
    "G4UIcmdWithADouble": "double",
    "G4UIcmdWithAString": "string",
}


def _normalize_text(text: str) -> str:
    """Collapse whitespace so C++ string fragments become readable prose."""
    return " ".join(text.split())


def _extract_joined_cpp_strings(argument_body: str) -> str:
    """Extract and join C++ string literals from calls like SetGuidance("a" "b")."""
    parts = re.findall(r'"((?:\\.|[^"\\])*)"', argument_body, flags=re.S)
    if not parts:
        return ""
    joined = " ".join(parts)
    # Keep this conservative: only normalize whitespace.
    return _normalize_text(joined)


def _find_section_for_position(section_markers: list[tuple[int, str]], position: int) -> str:
    """Return the most recent ADEPT_DOCS_SECTION marker before a command."""
    current = "Uncategorized"
    for marker_pos, section_name in section_markers:
        if marker_pos <= position:
            current = section_name
        else:
            break
    return current


def parse_messenger_commands(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Parse command metadata from AdePTConfigurationMessenger.cc.

    Parsing is intentionally lightweight/regex-based:
    - creation_re identifies command creation (type + UI path + variable name)
    - guidance/range/parameter regexes enrich those command rows
    - section markers group commands for output ordering
    """
    text = path.read_text(encoding="utf-8")

    # NOTE: This expects commands to be created as:
    #   fCmd = std::make_unique<G4UIcmdWith...>("/adept/...", this);
    # Keep this pattern stable or update the regex.
    creation_re = re.compile(
        r"^\s*(f\w+)\s*=\s*std::make_unique<([^>]+)>\(\"(/adept/[^\"]+)\",\s*this\);",
        flags=re.M,
    )
    guidance_re = re.compile(r"(f\w+)->SetGuidance\((.*?)\);", flags=re.S)
    range_re = re.compile(r'(f\w+)->SetRange\("([^"]+)"\);')
    pname_re = re.compile(r'(f\w+)->SetParameterName\("([^"]+)",\s*(?:true|false)\);')
    section_re = re.compile(r"^\s*//\s*ADEPT_DOCS_SECTION:\s*(.*?)\s*$", flags=re.M)

    section_markers = [(m.start(), _normalize_text(m.group(1))) for m in section_re.finditer(text)]
    section_order: list[str] = []
    for _, section_name in section_markers:
        if section_name not in section_order:
            section_order.append(section_name)

    commands: list[dict[str, str]] = []
    # Fast lookup so later SetGuidance/SetRange/SetParameterName calls can
    # update the row associated with a command variable.
    by_var: dict[str, dict[str, str]] = {}

    for m in creation_re.finditer(text):
        var, cls, cmd = m.groups()
        section = _find_section_for_position(section_markers, m.start())
        row = {
            "var": var,
            "class": cls,
            "command": cmd,
            "value_type": TYPE_MAP.get(cls, cls),
            "description": "",
            "range": "",
            "parameter": "",
            "section": section,
        }
        commands.append(row)
        by_var[var] = row

    for m in guidance_re.finditer(text):
        var, body = m.groups()
        if var in by_var:
            by_var[var]["description"] = _extract_joined_cpp_strings(body)

    for m in range_re.finditer(text):
        var, value = m.groups()
        if var in by_var:
            by_var[var]["range"] = value

    for m in pname_re.finditer(text):
        var, value = m.groups()
        if var in by_var:
            by_var[var]["parameter"] = value

    return commands, section_order


def parse_ci_template(path: Path) -> dict[str, str]:
    """Extract one example invocation per /adept command from CI macro."""
    command_to_example: dict[str, str] = {}

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if line.startswith("/adept/"):
            cmd = line.split()[0]
            # Keep the first occurrence to preserve a stable, concise example.
            command_to_example.setdefault(cmd, line)
            continue

    return command_to_example


def _escape_cell(value: str) -> str:
    return value.replace("|", r"\|")


def _format_command_rows(
    commands: list[dict[str, str]],
    examples: dict[str, str],
    section_order: list[str],
) -> str:
    """Render grouped command metadata into markdown tables."""
    grouped: "OrderedDict[str, list[dict[str, str]]]" = OrderedDict()

    for cmd in commands:
        section = cmd.get("section", "Uncategorized")
        grouped.setdefault(section, []).append(cmd)

    ordered_grouped: "OrderedDict[str, list[dict[str, str]]]" = OrderedDict()
    for section in section_order:
        if section in grouped:
            ordered_grouped[section] = grouped.pop(section)

    # Preserve source-order groups not covered by section markers.
    for section, rows in grouped.items():
        ordered_grouped[section] = rows

    chunks: list[str] = []
    for section, rows in ordered_grouped.items():
        chunks.append(f"## {section}\n")
        chunks.append("| Command | Value type | Description | Constraints | Example |\n")
        chunks.append("| --- | --- | --- | --- | --- |\n")

        for row in rows:
            cmd = f"`{row['command']}`"
            value_type = _escape_cell(row["value_type"] or "-")
            description = _escape_cell(row["description"] or "-")

            constraints_bits = []
            if row["parameter"]:
                constraints_bits.append(f"parameter `{row['parameter']}`")
            if row["range"]:
                constraints_bits.append(f"range `{row['range']}`")
            constraints = _escape_cell("; ".join(constraints_bits) if constraints_bits else "-")

            example_raw = examples.get(row["command"], "")
            example = f"`{_escape_cell(example_raw)}`" if example_raw else "-"

            chunks.append(f"| {cmd} | {value_type} | {description} | {constraints} | {example} |\n")
        chunks.append("\n")

    return "".join(chunks)


def generate(output_path: Path, messenger_path: Path, template_path: Path) -> dict[str, object]:
    """Generate runtime-parameters.md and return diagnostics for CI checks."""
    commands, section_order = parse_messenger_commands(messenger_path)
    examples = parse_ci_template(template_path)

    source_commands = {c["command"] for c in commands}
    template_commands = set(examples)
    unknown_template_commands = sorted(template_commands - source_commands)
    # Soft warnings here; main() decides whether to fail CI in strict mode.
    if not commands:
        print("WARNING: no /adept/ commands parsed from messenger source")

    if unknown_template_commands:
        print("WARNING: commands present in template but not in messenger source:")
        for cmd in unknown_template_commands:
            print(f"  - {cmd}")

    uncategorized = [c["command"] for c in commands if c.get("section") == "Uncategorized"]
    if uncategorized:
        print("WARNING: commands missing ADEPT_DOCS_SECTION marker:")
        for cmd in uncategorized:
            print(f"  - {cmd}")

    header = (
        "<!--\n"
        "SPDX-FileCopyrightText: 2026 CERN\n"
        "SPDX-License-Identifier: CC-BY-4.0\n"
        "-->\n\n"
        "# Runtime Parameters\n\n"
        "<!--\n"
        "This file is auto-generated.\n"
        "Do not edit manually.\n"
        "-->\n\n"
        "This page is generated from:\n\n"
        "- `src/AdePTConfigurationMessenger.cc` (authoritative command definitions and guidance)\n"
        "- `test/regression/scripts/test_ui_commands_template.mac` (example invocations used in CI)\n\n"
        "Regenerate with:\n\n"
        "```console\n"
        "python3 docs/scripts/generate_runtime_parameters.py\n"
        "```\n\n"
        "Categories are sourced from `// ADEPT_DOCS_SECTION: ...` markers in\n"
        "`src/AdePTConfigurationMessenger.cc`.\n\n"
        "Examples come from the CI UI command macro. If a command shows `-` in the\n"
        "Example column, add an invocation in `test_ui_commands_template.mac`.\n\n"
    )

    content = header + _format_command_rows(commands, examples, section_order)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")

    return {
        "command_count": len(commands),
        "unknown_template_commands": unknown_template_commands,
        "uncategorized_commands": uncategorized,
    }


def main() -> int:
    """Entry point used by local generation and CI."""
    docs_dir = Path(__file__).resolve().parents[1]
    repo_root = docs_dir.parent

    messenger_path = repo_root / "src" / "AdePTConfigurationMessenger.cc"
    template_path = repo_root / "test" / "regression" / "scripts" / "test_ui_commands_template.mac"
    output_path = docs_dir / "user" / "runtime-parameters.md"

    diagnostics = generate(output_path, messenger_path, template_path)
    print(f"Generated {output_path}")

    # Keep CI strict by default: generator must stay aligned with source + macro.
    strict_failures = []
    if diagnostics["command_count"] == 0:
        strict_failures.append("no commands parsed from messenger source")
    if diagnostics["unknown_template_commands"]:
        strict_failures.append("template contains commands not found in messenger source")
    if diagnostics["uncategorized_commands"]:
        strict_failures.append("some commands are missing ADEPT_DOCS_SECTION markers")

    if strict_failures:
        print("ERROR: runtime-parameters generation failed strict validation:")
        for failure in strict_failures:
            print(f"  - {failure}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
