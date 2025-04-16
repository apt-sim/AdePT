#! /usr/bin/env python3

# SPDX-FileCopyrightText: 2025 CERN
# SPDX-License-Identifier: Apache-2.0

import argparse
import os


def generate_macro(template_path, output_path, gdml_name, num_threads, num_events, num_trackslots, num_leakslots, num_hitslots, gun_type, event_file, track_in_all_regions, regions):
    # Read the template file
    with open(template_path, 'r') as template_file:
        template_content = template_file.read()

    # Substitute placeholders with actual values
    macro_content = template_content.replace("$gdml_name", gdml_name)
    macro_content = macro_content.replace("$num_threads", str(num_threads))
    macro_content = macro_content.replace("$num_events", str(num_events))
    macro_content = macro_content.replace("$num_trackslots", str(num_trackslots))
    macro_content = macro_content.replace("$num_leakslots", str(num_leakslots))
    macro_content = macro_content.replace("$num_hitslots", str(num_hitslots))
    macro_content = macro_content.replace("$gun_type", str(gun_type))
    macro_content = macro_content.replace(
        "$track_in_all_regions", str(track_in_all_regions))

    if (str(gun_type) == "hepmc"):
        hepmc_part = "/generator/hepmcAscii/maxevents 256 \n\
                      /generator/hepmcAscii/firstevent 0 \n\
                      /generator/hepmcAscii/open $event_file \n\
                      /generator/hepmcAscii/verbose 0"
        macro_content = macro_content.replace("$hepmc_part", hepmc_part)
        macro_content = macro_content.replace("$event_file", str(event_file))
    else:
        macro_content = macro_content.replace("$hepmc_part", str(""))

    # Regions should be a comma-separated list of region names
    region_part = []
    for i in regions.split(","):
        region = i.strip()
        if region:  # Empty regions list or empty region name
            region_part.append(f"/adept/addGPURegion {region}")
    region_part = "\n".join(region_part)
    macro_content = macro_content.replace("$regions", region_part)

    # Write the output macro file
    with open(output_path, 'w') as output_file:
        output_file.write(macro_content)

    print(f"Macro file generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Geant4 macro file from template.")
    parser.add_argument("--template", required=True,
                        help="Path to the macro template file.")
    parser.add_argument("--output", required=True,
                        help="Path to save the generated macro file.")
    parser.add_argument("--gdml_name", required=True,
                        help="Path to the GDML geometry file.")
    parser.add_argument("--num_threads", type=int, required=True,
                        help="Number of threads to use.")
    parser.add_argument("--num_events", type=int, required=True,
                        help="Number of events to simulate.")
    parser.add_argument("--num_trackslots", type=int, default=12,
                        help="Number of trackslots in million. Should be chosen according to the GPU memory")
    parser.add_argument("--num_leakslots", type=float, default=12,
                        help="Number of leakslots in million. Should be chosen according to the GPU memory")
    parser.add_argument("--num_hitslots", type=int, default=12,
                        help="Number of hitslots in million. Should be chosen according to the GPU memory")
    parser.add_argument("--gun_type", default="setDefault",
                        help="Type of particle gun. Must be 'hepmc' or 'setDefault'.")
    parser.add_argument("--event_file", help="Path to the hepmc3 event file")
    parser.add_argument("--track_in_all_regions", required=False,
                        default="True", help="True or False")
    parser.add_argument("--regions", type=str, required=False, default="",
                        help="Comma-separated list of regions in which to do GPU transport, only if track_in_all_regions is False")
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
        gdml_name=args.gdml_name,
        num_threads=args.num_threads,
        num_events=args.num_events,
        num_trackslots=args.num_trackslots,
        num_leakslots=args.num_leakslots,
        num_hitslots=args.num_hitslots,
        gun_type=args.gun_type,
        event_file=args.event_file,
        track_in_all_regions=args.track_in_all_regions,
        regions=args.regions
    )


if __name__ == "__main__":
    main()
