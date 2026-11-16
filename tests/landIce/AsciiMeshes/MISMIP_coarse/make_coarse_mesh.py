#!/usr/bin/env python3
"""
Coarsen the MISMIP 2D test mesh (../MISMIP/mismip_2d.exo) and its nodal fields.

The MISMIP mesh is a lattice of equilateral triangles: rows of nodes at y = j*dy (j = 0..ny),
with spacing a = 2*dy/sqrt(3) along each row, and the rows alternately shifted by a/2. The
coarse mesh is the same kind of lattice, with m times the spacing, built on every m-th row of
the fine mesh, and on nodes of the fine mesh only. So the nodal fields of the coarse mesh are
the values of the fine fields at the same nodes: no interpolation.

The node sets, side sets, names and ids follow the fine mesh:
  node sets: 1 "node_set" (all nodes), 2 "boundary_node_set_0" (empty),
             3 "boundary_node_set_1" (first node of each row, x ~ 0),
             4 "boundary_node_set_2" (last node of each row, except the first and last rows),
             5 "boundary_node_set_3" (first and last rows, except their first node);
  side sets: 1 "boundary_side_set" (all boundary edges),
             2 "boundary_side_set_1" (all boundary edges, except those of the last nodes),
             3 "boundary_side_set_2" (edges between the last nodes of consecutive rows).
Running with -m 1 rebuilds the fine mesh, which the script compares with the original.

Usage: make_coarse_mesh.py --fine-dir ../MISMIP --out-dir . -m 3
Requires numpy and netCDF4.
"""

import argparse
import os
import sys

import netCDF4
import numpy as np

SCALAR_FIELDS = ["thickness.ascii", "surface_height.ascii", "bed_topography.ascii"]
VECTOR_FIELDS = ["averaged_velocity.ascii"]


def read_fine_mesh(fname):
    f = netCDF4.Dataset(fname)
    f.set_auto_mask(False)
    mesh = {
        "x": f["coordx"][:].copy(),
        "y": f["coordy"][:].copy(),
        "connect": f["connect1"][:].copy() - 1,
        "node_num_map": f["node_num_map"][:].copy(),
        "ns": {},
        "ss": {},
    }
    for k in range(1, len(f["ns_prop1"][:]) + 1):
        name = "node_ns%d" % k
        mesh["ns"][k] = f[name][:].copy() - 1 if name in f.variables else np.zeros(0, dtype=int)
    for k in range(1, len(f["ss_prop1"][:]) + 1):
        mesh["ss"][k] = (f["elem_ss%d" % k][:].copy() - 1, f["side_ss%d" % k][:].copy())
    f.close()
    if not np.array_equal(mesh["node_num_map"], np.arange(1, len(mesh["x"]) + 1)):
        sys.exit("Error: the node ids of the fine mesh are not 1..N (the ascii fields are ordered by id).")
    return mesh


def lattice_rows(x, y, tol=1e-6):
    """Rows of the fine mesh: list of arrays of node indices, sorted by x, for increasing y."""
    ys = np.unique(np.round(y / tol) * tol)
    rows = []
    for yy in ys:
        idx = np.where(np.abs(y - yy) < 10 * tol)[0]
        rows.append(idx[np.argsort(x[idx])])
    return rows


def coarse_rows(fine_rows, x, m):
    """Rows of the coarse mesh, as indices of fine nodes."""
    nrows = len(fine_rows) - 1
    if nrows % m != 0:
        sys.exit("Error: the number of row intervals of the fine mesh (%d) is not a multiple of %d." % (nrows, m))
    a = np.median(np.diff(x[fine_rows[0]]))
    # Shift of the first node of each coarse row, w.r.t. the leftmost node of the fine rows:
    # the coarse rows alternate as the fine rows do (the first starts a_c/2 = m*a/2 to the right
    # of the second), keeping the leftmost node of the fine mesh (x = 0) in the rows that have it.
    xmin = min(x[r[0]] for r in fine_rows)
    rows = []
    for J in range(nrows // m + 1):
        frow = fine_rows[J * m]
        x0 = xmin + (m * a / 2.0 if J % 2 == 0 else 0.0)
        # Every m-th fine node, starting from the fine node at x0
        start = np.argmin(np.abs(x[frow] - x0))
        if abs(x[frow[start]] - x0) > 1e-6 * a:
            sys.exit("Error: no fine node at x = %g in row %d." % (x0, J * m))
        rows.append(frow[start::m])
    return rows


def triangulate(rows, x):
    """Triangles between consecutive rows (counterclockwise), and the boundary edges by kind."""
    tris = []
    for J in range(len(rows) - 1):
        lo, up = rows[J], rows[J + 1]
        i = k = 0
        while i < len(lo) - 1 or k < len(up) - 1:
            advance_lower = k == len(up) - 1 or (i < len(lo) - 1 and x[lo[i + 1]] <= x[up[k + 1]])
            if advance_lower:
                tris.append((lo[i], lo[i + 1], up[k]))
                i += 1
            else:
                tris.append((lo[i], up[k + 1], up[k]))
                k += 1
    return np.array(tris, dtype=int)


def side_sets(tris, rows):
    """Elements and local sides (Exodus: 1 = nodes 1-2, 2 = 2-3, 3 = 3-1) of the boundary edges."""
    local = {1: (0, 1), 2: (1, 2), 3: (2, 0)}
    edge_owner = {}
    count = {}
    for e, t in enumerate(tris):
        for s, (p, q) in local.items():
            key = tuple(sorted((t[p], t[q])))
            count[key] = count.get(key, 0) + 1
            edge_owner[key] = (e, s)
    boundary = [key for key, c in count.items() if c == 1]
    last = set(r[-1] for r in rows)
    east = [key for key in boundary if key[0] in last and key[1] in last]
    other = [key for key in boundary if key not in set(east)]

    def as_ss(keys):
        es = sorted(edge_owner[key] for key in keys)
        return np.array([e for e, _ in es], dtype=int), np.array([s for _, s in es], dtype=int)

    return {1: as_ss(boundary), 2: as_ss(other), 3: as_ss(east)}


def node_sets(rows, nnodes):
    first = [r[0] for r in rows]
    last_inner = [r[-1] for r in rows[1:-1]]
    lateral = list(rows[0][1:]) + list(rows[-1][1:])
    return {1: np.arange(nnodes), 2: np.zeros(0, dtype=int),
            3: np.array(first), 4: np.array(last_inner), 5: np.array(lateral)}


def build(fine, m):
    x, y = fine["x"], fine["y"]
    rows_f = lattice_rows(x, y)
    rows_c = coarse_rows(rows_f, x, m)
    # Coarse nodes, numbered row by row (as in the fine mesh)
    fine_ids = np.concatenate(rows_c)
    new_id = {fid: i for i, fid in enumerate(fine_ids)}
    rows = [np.array([new_id[i] for i in r]) for r in rows_c]
    xc, yc = x[fine_ids], y[fine_ids]
    tris = triangulate(rows, xc)
    # Counterclockwise orientation
    p = np.stack([xc, yc], 1)
    d1, d2 = p[tris[:, 1]] - p[tris[:, 0]], p[tris[:, 2]] - p[tris[:, 0]]
    area = 0.5 * (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    tris[area < 0] = tris[area < 0][:, [0, 2, 1]]
    return {"x": xc, "y": yc, "connect": tris, "fine_ids": fine_ids,
            "ns": node_sets(rows, len(xc)), "ss": side_sets(tris, rows)}


def compare_with_fine(fine, rebuilt):
    """With m = 1, the rebuilt mesh must be the fine mesh (up to the order of the elements)."""
    ok = np.array_equal(rebuilt["fine_ids"], np.arange(len(fine["x"])))
    tri = lambda c: set(tuple(sorted(t)) for t in c)
    ok &= tri(rebuilt["connect"]) == tri(fine["connect"])
    for k in fine["ns"]:
        ok &= set(rebuilt["ns"][k]) == set(fine["ns"][k])
    local = {1: (0, 1), 2: (1, 2), 3: (2, 0)}
    edges = lambda mesh, k: set(tuple(sorted((mesh["connect"][e][local[s][0]], mesh["connect"][e][local[s][1]])))
                                for e, s in zip(*mesh["ss"][k]))
    for k in fine["ss"]:
        ok &= edges(rebuilt, k) == edges(fine, k)
    return ok


def write_exodus(fname, mesh, fine_fname, m):
    """Same layout as the fine mesh (written by IOSS): netCDF3 64-bit offset, same names."""
    src = netCDF4.Dataset(fine_fname)
    src.set_auto_mask(False)
    f = netCDF4.Dataset(fname, "w", format="NETCDF3_64BIT_OFFSET")
    nn, ne = len(mesh["x"]), len(mesh["connect"])
    ns, ss = mesh["ns"], mesh["ss"]

    info = ["MISMIP 2D test mesh coarsened by a factor %d from %s" % (m, os.path.basename(fine_fname)),
            "by make_coarse_mesh.py (tests/landIce/AsciiMeshes/MISMIP_coarse)."]
    f.createDimension("num_qa_rec", 1)
    f.createDimension("four", 4)
    f.createDimension("len_string", 33)
    f.createDimension("num_info", len(info))
    f.createDimension("len_line", 81)
    f.createDimension("len_name", 33)
    f.createDimension("num_dim", 2)
    f.createDimension("time_step", None)
    f.createDimension("num_nodes", nn)
    f.createDimension("num_elem", ne)
    f.createDimension("num_el_blk", 1)
    f.createDimension("num_node_sets", len(ns))
    f.createDimension("num_side_sets", len(ss))
    f.createDimension("num_el_in_blk1", ne)
    f.createDimension("num_nod_per_el1", 3)
    for k in sorted(ns):
        if len(ns[k]) > 0:
            f.createDimension("num_nod_ns%d" % k, len(ns[k]))
    for k in sorted(ss):
        f.createDimension("num_side_ss%d" % k, len(ss[k][0]))

    def chars(var, strings, length):
        arr = np.zeros((len(strings), length), dtype="S1")
        for i, s in enumerate(strings):
            b = s.encode()[: length - 1]
            arr[i, : len(b)] = np.frombuffer(b, dtype="S1")
        var[:] = arr

    v = f.createVariable("qa_records", "S1", ("num_qa_rec", "four", "len_string"))
    qa = np.zeros((1, 4, 33), dtype="S1")
    for i, s in enumerate(["make_coarse_mesh.py", "1.0", "", ""]):
        b = s.encode()
        qa[0, i, : len(b)] = np.frombuffer(b, dtype="S1")
    v[:] = qa
    chars(f.createVariable("info_records", "S1", ("num_info", "len_line")), info, 81)
    f.createVariable("time_whole", "f8", ("time_step",))[0] = 0.0
    f.createVariable("node_num_map", "i4", ("num_nodes",))[:] = np.arange(1, nn + 1)
    f.createVariable("elem_num_map", "i4", ("num_elem",))[:] = np.arange(1, ne + 1)
    f.createVariable("eb_status", "i4", ("num_el_blk",))[:] = [1]
    v = f.createVariable("eb_prop1", "i4", ("num_el_blk",)); v.setncattr("name", "ID"); v[:] = [1]
    chars(f.createVariable("eb_names", "S1", ("num_el_blk", "len_name")), ["Element Block 0"], 33)
    f.createVariable("ns_status", "i4", ("num_node_sets",))[:] = [1 if len(ns[k]) > 0 else 0 for k in sorted(ns)]
    v = f.createVariable("ns_prop1", "i4", ("num_node_sets",)); v.setncattr("name", "ID"); v[:] = sorted(ns)
    chars(f.createVariable("ns_names", "S1", ("num_node_sets", "len_name")),
          ["node_set", "boundary_node_set_0", "boundary_node_set_1", "boundary_node_set_2", "boundary_node_set_3"], 33)
    f.createVariable("ss_status", "i4", ("num_side_sets",))[:] = [1] * len(ss)
    v = f.createVariable("ss_prop1", "i4", ("num_side_sets",)); v.setncattr("name", "ID"); v[:] = sorted(ss)
    chars(f.createVariable("ss_names", "S1", ("num_side_sets", "len_name")),
          ["boundary_side_set", "boundary_side_set_1", "boundary_side_set_2"], 33)
    f.createVariable("coordx", "f8", ("num_nodes",))[:] = mesh["x"]
    f.createVariable("coordy", "f8", ("num_nodes",))[:] = mesh["y"]
    chars(f.createVariable("coor_names", "S1", ("num_dim", "len_name")), ["x", "y"], 33)
    v = f.createVariable("connect1", "i4", ("num_el_in_blk1", "num_nod_per_el1"))
    v.elem_type = "tri3"
    v[:] = mesh["connect"] + 1
    for k in sorted(ns):
        if len(ns[k]) > 0:
            f.createVariable("node_ns%d" % k, "i4", ("num_nod_ns%d" % k,))[:] = np.sort(ns[k]) + 1
            f.createVariable("dist_fact_ns%d" % k, "f8", ("num_nod_ns%d" % k,))[:] = 1.0
    for k in sorted(ss):
        f.createVariable("elem_ss%d" % k, "i4", ("num_side_ss%d" % k,))[:] = ss[k][0] + 1
        f.createVariable("side_ss%d" % k, "i4", ("num_side_ss%d" % k,))[:] = ss[k][1]

    for att in ["api_version", "version"]:
        f.setncattr(att, np.float32(src.getncattr(att)))
    f.floating_point_word_size = np.int32(8)
    f.file_size = np.int32(1)
    f.maximum_name_length = np.int32(19)
    f.int64_status = np.int32(0)
    f.title = "MISMIP 2D test mesh, coarsened by a factor %d" % m
    f.last_written_time = 0.0
    f.close()
    src.close()


def read_ascii(fname):
    with open(fname) as fh:
        header = fh.readline().split()
        values = np.array(fh.read().split(), dtype=float)
    return header, values


def write_fields(fine_dir, out_dir, fine_ids, nfine):
    for name in SCALAR_FIELDS + VECTOR_FIELDS:
        path = os.path.join(fine_dir, name)
        if not os.path.exists(path):
            continue
        header, values = read_ascii(path)
        ncomp = int(header[1]) if len(header) > 1 else 1
        if int(header[0]) != nfine or len(values) != nfine * ncomp:
            sys.exit("Error: unexpected size of %s." % path)
        # Component-major, as read by Albany (readVectorFileSerial)
        comps = values.reshape(ncomp, nfine)[:, fine_ids]
        with open(os.path.join(out_dir, name), "w") as fh:
            fh.write(" ".join([str(len(fine_ids))] + header[1:]) + "\n")
            for c in comps:
                for v in c:
                    fh.write("%.15g\n" % v)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fine-dir", default="../MISMIP")
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("-m", type=int, default=3, help="coarsening factor (divides the number of rows - 1)")
    args = parser.parse_args()

    fine_fname = os.path.join(args.fine_dir, "mismip_2d.exo")
    fine = read_fine_mesh(fine_fname)

    rebuilt = build(fine, 1)
    if not compare_with_fine(fine, rebuilt):
        sys.exit("Error: the lattice rebuilt with m = 1 is not the fine mesh.")
    print("Check: the lattice rebuilt with m = 1 is the fine mesh.")

    coarse = build(fine, args.m)
    os.makedirs(args.out_dir, exist_ok=True)
    write_exodus(os.path.join(args.out_dir, "mismip_2d.exo"), coarse, fine_fname, args.m)
    write_fields(args.fine_dir, args.out_dir, coarse["fine_ids"], len(fine["x"]))
    print("Coarse mesh (m = %d): %d nodes, %d triangles; node sets %s; side sets %s"
          % (args.m, len(coarse["x"]), len(coarse["connect"]),
             [len(coarse["ns"][k]) for k in sorted(coarse["ns"])],
             [len(coarse["ss"][k][0]) for k in sorted(coarse["ss"])]))


if __name__ == "__main__":
    main()
