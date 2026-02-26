#! /usr/bin/env python3

# SPDX-FileCopyrightText: 2025 CERN
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import re


def generate_macro(template_path, output_path, args):
    # Read the template file
    with open(template_path, 'r') as template_file:
        template_content = template_file.read()

    # Validate required placeholders based on template
    placeholders = set(re.findall(r"\$(\w+)", template_content))

    args_dict = vars(args)

    missing = [
        p for p in placeholders
        if p in args_dict and args_dict[p] is None
    ]
    if missing:
        raise ValueError(
            f"Missing required arguments for template: {', '.join(missing)}"
        )

    # Original replacement logic

    macro_content = template_content.replace("$gdml_name", str(args.gdml_name))

    macro_content = macro_content.replace("$num_threads", str(args.num_threads))
    macro_content = macro_content.replace("$num_events", str(args.num_events))
    macro_content = macro_content.replace("$num_trackslots", str(args.num_trackslots))
    macro_content = macro_content.replace("$num_leakslots", str(args.num_leakslots))
    macro_content = macro_content.replace("$num_hitslots", str(args.num_hitslots))
    macro_content = macro_content.replace("$adept_seed", str(args.adept_seed))
    macro_content = macro_content.replace("$gun_type", str(args.gun_type))
    macro_content = macro_content.replace("$gun_number", str(args.gun_number))
    macro_content = macro_content.replace("$track_in_all_regions", str(args.track_in_all_regions))

    if str(args.gun_type) == "hepmc":
        hepmc_part = "/generator/hepmcAscii/maxevents 256 \n\
                      /generator/hepmcAscii/firstevent 0 \n\
                      /generator/hepmcAscii/open $event_file \n\
                      /generator/hepmcAscii/verbose 0"
        macro_content = macro_content.replace("$hepmc_part", hepmc_part)
        macro_content = macro_content.replace("$event_file", str(args.event_file))
    else:
        macro_content = macro_content.replace("$hepmc_part", str(""))

    # Regions should be a comma-separated list of region names
    region_part = []
    for i in args.regions.split(","):
        region = i.strip()
        if region:  # Empty regions list or empty region name
            region_part.append(f"/adept/addGPURegion {region}")
    region_part = "\n".join(region_part)
    macro_content = macro_content.replace("$regions", region_part)

    # Woodcock tracking regions should be a comma-separated list of region names
    wdt_region_part = []
    for i in args.wdt_regions.split(","):
        wdt_region = i.strip()
        if wdt_region:  # Empty wdt regions list or empty region name
            wdt_region_part.append(f"/adept/addWDTRegion {wdt_region}")
    wdt_region_part = "\n".join(wdt_region_part)
    macro_content = macro_content.replace("$wdt_regions", wdt_region_part)

    # Write the output macro file
    with open(output_path, 'w') as output_file:
        output_file.write(macro_content)

    print(f"Macro file generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Geant4 macro file from template.")
    parser.add_argument("--template", help="Path to the macro template file.")
    parser.add_argument("--output", help="Path to save the generated macro file.")
    parser.add_argument("--gdml_name", help="Path to the GDML geometry file.")
    parser.add_argument("--num_threads", type=int, help="Number of threads to use.")
    parser.add_argument("--num_events", type=int, help="Number of events to simulate.")
    parser.add_argument("--num_trackslots", type=int, default=12,
                        help="Number of trackslots in million. Should be chosen according to the GPU memory")
    parser.add_argument("--num_leakslots", type=float, default=12,
                        help="Number of leakslots in million. Should be chosen according to the GPU memory")
    parser.add_argument("--num_hitslots", type=int, default=12,
                        help="Number of hitslots in million. Should be chosen according to the GPU memory")
    parser.add_argument("--adept_seed", type=int, default=1234567,
                        help="Base seed for AdePT RNG (default: 1234567)")
    parser.add_argument("--gun_type", default="setDefault",
                        help="Type of particle gun. Must be 'hepmc' or 'setDefault'.")
    parser.add_argument("--gun_number", type=int, default=100,
                        help="Number of primary particles per event (default: 100)")
    parser.add_argument("--event_file", help="Path to the hepmc3 event file")
    parser.add_argument("--track_in_all_regions", default="True", help="True or False")
    parser.add_argument("--regions", type=str, required=False, default="",
                        help="Comma-separated list of regions in which to do GPU transport, only if track_in_all_regions is False")
    parser.add_argument("--wdt_regions", type=str, required=False, default="",
                        help="Comma-separated list of Woodcock tracking regions")
    args = parser.parse_args()

    if str(args.gun_type) not in ["hepmc", "setDefault"]:
        print("Error: --gun_type must either be 'hepmc' or 'setDefault'.")
        exit(1)

    # Validate template file path

    if not os.path.exists(args.template):
        print(f"Error: Template file not found: {args.template}")
        exit(1)

    # Generate the macro file
    generate_macro(
        template_path=args.template,
        output_path=args.output,
        args=args
    )


if __name__ == "__main__":
    main()
