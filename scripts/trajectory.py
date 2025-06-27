# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0
# Author: ChatGPT o4-mini-high 27 Jun 2025
import re
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
import csv

"""
Trajectory V3 - Field Step Point Parser with Debug Counters

This script parses a particle trajectory log and plots:
 - Step points colored by kinetic energy
 - Field step points (lines containing 'field_point') in small markers
 - Straight lines between successive step points
 - Quadratic spline connections when field steps are involved
 - Direction vectors scaled by a fraction of the trajectory span
 - Optionally forces equal x/y/z aspect ratio in the 3D plot

It also prints a debug line for each point it reads:
  counter, file-line, type (step|field|linear), pos={x,y,z}

Input parameters:
  --min-step    : Minimum step number to include (default: all)
  --max-step    : Maximum step number to include (default: all)
  --dir-fraction: Fraction of trajectory span to scale direction vectors (default: 0.01)
  --input-file  : Path to input log file (default: 'track_log.txt')
  --output-csv  : Path to output CSV file (default: 'parsed_trajectory.csv')
  --show-labels : Show navigation/step labels on the plot
  --equal-aspect  : Force equal aspect ratio (x:y:z = 1:1:1)
"""

def get_args():
    parser = argparse.ArgumentParser(
        description='Plot particle trajectory with field steps using quadratic splines.'
    )
    parser.add_argument('--min-step',    type=int,   default=None, help='Minimum step number to include')
    parser.add_argument('--max-step',    type=int,   default=None, help='Maximum step number to include')
    parser.add_argument('--dir-fraction',type=float, default=0.01, help='Fraction of trajectory span for vectors')
    parser.add_argument('--input-file',  type=str,   default='track_log.txt', help='Input log filename')
    parser.add_argument('--output-csv',  type=str,   default='parsed_trajectory.csv', help='Output CSV filename')
    parser.add_argument('--show-labels', action='store_true', help='Show navigation/step labels')
    parser.add_argument('--equal-aspect', action='store_true', help='Force equal x/y aspect ratio (ignores z)')
    return parser.parse_args()

# Parse arguments
args         = get_args()
min_step     = args.min_step
max_step     = args.max_step
fraction     = args.dir_fraction
input_file   = args.input_file
output_csv   = args.output_csv
show_labels  = args.show_labels
equal_aspect = args.equal_aspect

# Regex patterns
header_re      = re.compile(r"^== evt \d+.*? step (?P<step>\d+) ekin (?P<ekin>[0-9.]+) MeV", re.MULTILINE)
posdir_re      = re.compile(r"pos \{(?P<px>-?[0-9.eE+-]+), (?P<py>-?[0-9.eE+-]+), (?P<pz>-?[0-9.eE+-]+)\} dir \{(?P<dx>-?[0-9.eE+-]+), (?P<dy>-?[0-9.eE+-]+), (?P<dz>-?[0-9.eE+-]+)\}")
field_point_re = re.compile(
    r"field_point:.*?pos\s*\{(?P<fx>-?[0-9.eE+-]+),\s*(?P<fy>-?[0-9.eE+-]+),\s*(?P<fz>-?[0-9.eE+-]+)\}"
    r".*?dir\s*\{(?P<fdx>-?[0-9.eE+-]+),\s*(?P<fdy>-?[0-9.eE+-]+),\s*(?P<fdz>-?[0-9.eE+-]+)\}",
    re.IGNORECASE
)

# Read full log
with open(input_file, 'r') as f:
    txt = f.read()
headers = list(header_re.finditer(txt))
bounds  = [m.start() for m in headers] + [len(txt)]

# Prepare
point_counter = 0
steps_data    = []  # (step, ekin, (px,py,pz), (dx,dy,dz))
field_pts     = []
field_dirs    = []
owners        = []
field_types   = []

# Parse
for i, h in enumerate(headers):
    step = int(h.group('step'))
    block = txt[bounds[i]:bounds[i+1]]
    mpos  = posdir_re.search(block)
    px, py, pz = map(float, (mpos.group('px'), mpos.group('py'), mpos.group('pz')))
    dx, dy, dz = map(float, (mpos.group('dx'), mpos.group('dy'), mpos.group('dz')))
    # Debug step
    header_line = txt[:h.start()].count('\n') + 1
    print(f"{point_counter} {header_line} step pos={{ {px}, {py}, {pz} }}")
    point_counter += 1
    # Range check
    if (min_step is not None and step < min_step) or (max_step is not None and step > max_step):
        continue
    ekin = float(h.group('ekin'))
    steps_data.append((step, ekin, (px,py,pz), (dx,dy,dz)))
    # Field points
    lines = block.splitlines()
    for idx_line, line in enumerate(lines):
        if 'field_point:' not in line:
            continue
        mf = field_point_re.search(line)
        if not mf:
            continue
        fx, fy, fz   = map(float, (mf.group('fx'), mf.group('fy'), mf.group('fz')))
        fdx, fdy, fdz = map(float, (mf.group('fdx'), mf.group('fdy'), mf.group('fdz')))
        # detect 'linear' between this field_point and next field_point or end of block
        subsequent = lines[idx_line+1:]
        end_idx = len(subsequent)
        for k, ln in enumerate(subsequent):
            if 'field_point:' in ln:
                end_idx = k
                break
        segment = subsequent[:end_idx]
        is_linear = any('linear' in ln for ln in segment)
        # debug print
        abs_line = txt[:bounds[i]].count('\n') + idx_line + 1
        kind     = 'linear' if is_linear else 'field'
        print(f"{point_counter} {abs_line} {kind} pos={{ {fx}, {fy}, {fz} }}")
        point_counter += 1
        # store if not duplicate
        if (fx,fy,fz) != (px,py,pz):
            field_pts.append((fx,fy,fz))
            field_dirs.append((fdx,fdy,fdz))
            owners.append(len(steps_data)-1)
            field_types.append('step' if is_linear else 'field')
# Convert to numpy arrays
step_pts   = np.array([pos for *_ ,pos,_ in steps_data])
ekins      = np.array([e   for *_ ,e,_,_ in steps_data])
field_pts  = np.array(field_pts)  if field_pts  else np.empty((0,3))
field_dirs = np.array(field_dirs) if field_dirs else np.empty((0,3))
owners     = np.array(owners)     if owners     else np.empty((0,),int)

# Arrow length
if len(step_pts) > 1:
    span = np.linalg.norm(step_pts[-1] - step_pts[0])
    al   = span * fraction
else:
    al = 1.0

# Color mapping
cmap      = plt.get_cmap('viridis')
norm      = plt.Normalize(ekins.min(), ekins.max())
step_cols = cmap(norm(ekins))

# Build ordered sequence
ordered, types, colors = [], [], []
for idx, (_,_,pt,_) in enumerate(steps_data):
    ordered.append(pt);      types.append('step');  colors.append(step_cols[idx])
    mask = np.where(np.array(owners)==idx)[0]
    for j in mask:
        p     = field_pts[j]
        ftype = field_types[j]
        ordered.append(p); types.append(ftype); colors.append(step_cols[idx])
ordered = np.array(ordered)
colors  = np.array(colors)

# Plot
fig = plt.figure(figsize=(10,8))
ax  = fig.add_subplot(111, projection='3d')
sc  = ax.scatter(step_pts[:,0], step_pts[:,1], step_pts[:,2], c=step_cols, s=20)
fig.colorbar(sc, ax=ax, pad=0.1).set_label('Kinetic Energy (MeV)')
if field_pts.size:
    ax.scatter(field_pts[:,0], field_pts[:,1], field_pts[:,2], c=step_cols[owners], s=0.2)

# Connect points
for j in range(len(ordered)-1):
    P0, P1 = ordered[j], ordered[j+1]
    if types[j:j+2] == ['step','step']:
        line = np.vstack([P0,P1])
        ax.plot(line[:,0],line[:,1],line[:,2],c='black',linestyle=':',linewidth=0.5)
    else:
        if j+2 < len(ordered):
            P2 = ordered[j+2]
            t  = np.linspace(0,1,20)
            L0 = 2*(t-0.5)*(t-1)
            L1 = -4*t*(t-1)
            L2 = 2*t*(t-0.5)
            curve = (L0[:,None]*P0 + L1[:,None]*P1 + L2[:,None]*P2)
            ax.plot(curve[:,0],curve[:,1],curve[:,2],c='darkgray',linestyle=':',linewidth=0.5)
# Equalize x and y axes if requested
if equal_aspect:
    # use all plotted points for extents
    coords_xy = np.vstack([step_pts[:,:2], field_pts[:,:2]]) if field_pts.size else step_pts[:,:2]
    x_min, x_max = coords_xy[:,0].min(), coords_xy[:,0].max()
    y_min, y_max = coords_xy[:,1].min(), coords_xy[:,1].max()
    x_center = 0.5 * (x_max + x_min)
    y_center = 0.5 * (y_max + y_min)
    radius   = 0.5 * max(x_max - x_min, y_max - y_min)
    ax.set_xlim(x_center - radius, x_center + radius)
    ax.set_ylim(y_center - radius, y_center + radius)
# Draw direction vectors
def draw_vec(p,d,col):
    v = np.array(d); n=np.linalg.norm(v)
    if n: v=v/n*al
    ax.plot([p[0],p[0]+v[0]], [p[1],p[1]+v[1]], [p[2],p[2]+v[2]], c=col, linewidth=0.5)

for idx, (_,_,pt,dirv) in enumerate(steps_data):
    draw_vec(pt, dirv, step_cols[idx])
for pt,dirv,own in zip(field_pts, field_dirs, owners):
    draw_vec(pt, dirv, step_cols[own])

ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_title(f"Trajectory (steps {min_step or 'all'} to {max_step or 'all'})")
plt.tight_layout(); plt.show(block=True)

# CSV output
with open(output_csv,'w',newline='') as cf:
    writer = csv.writer(cf)
    writer.writerow(['step','px','py','pz','ekin'])
    for s,e,pt,dirv in steps_data:
        writer.writerow([s,*pt,e])
