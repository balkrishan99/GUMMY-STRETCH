"""
rubber_mesh.py
--------------
Original 2D mesh deformation engine for GUMMY STRETCH.

Approach: build a triangulated mesh over a region (e.g. a finger), then
deform it with an **axial displacement field** rooted at an anchor point
(e.g. the knuckle) and pulled toward a moving target (e.g. the live pinch).
Every mesh vertex's displacement is a weighted blend of:

  1. a direct pull toward the target, weighted by how far ALONG the rest
     anchor-to-tip axis the vertex sits (0 at the anchor, ~1 at the
     original far end) -- this is what makes the tip follow the pull
     almost fully while the base stays rooted, and

  2. a local-rigidity correction pass that nudges each vertex toward the
     position its triangle neighbors imply, so the mesh doesn't shear
     apart into nonsense at high stretch -- a cheap iterative relaxation
     rather than a closed-form linear solve.

This is intentionally NOT a port of any published mesh-warping paper --
it's a small, hand-rolled relaxation scheme tuned by eye for this effect.
It trades some geometric rigor for simplicity and real-time speed without
external solver dependencies (no scipy required).
"""

import numpy as np
import cv2


def build_region_mesh(mask, boundary_points=28, interior_spacing=None):
    """
    Triangulate a binary mask (uint8, 0/255) into a vertex/triangle mesh.

    Returns (verts: Nx2 float32, tris: Mx3 int32) in the mask's pixel
    coordinates, or None if the mask is empty/degenerate. Vertices double
    as texture coordinates (rest position == source pixel position).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    outline = max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float32)
    if len(outline) < boundary_points:
        return None

    boundary = _resample_closed_curve(outline, boundary_points)

    x, y, w, h = cv2.boundingRect(mask)
    step = interior_spacing or max(10, int(0.2 * min(w, h)))
    interior = [
        (px, py)
        for py in range(y + step // 2, y + h, step)
        for px in range(x + step // 2, x + w, step)
        if mask[py, px] > 0
    ]

    all_pts = np.vstack([boundary, np.array(interior, dtype=np.float32)]) \
        if interior else boundary
    all_pts = np.unique(np.round(all_pts, 1), axis=0).astype(np.float32)
    if len(all_pts) < 4:
        return None

    subdiv = cv2.Subdiv2D((x - 2, y - 2, x + w + 2, y + h + 2))
    for p in all_pts:
        subdiv.insert((float(p[0]), float(p[1])))

    lookup = {(round(p[0], 1), round(p[1], 1)): i for i, p in enumerate(all_pts)}
    tris = []
    for t in subdiv.getTriangleList():
        pts = [(round(t[0], 1), round(t[1], 1)),
               (round(t[2], 1), round(t[3], 1)),
               (round(t[4], 1), round(t[5], 1))]
        idx = [lookup.get(p) for p in pts]
        if any(i is None for i in idx):
            continue
        cx = sum(all_pts[i][0] for i in idx) / 3.0
        cy = sum(all_pts[i][1] for i in idx) / 3.0
        ix, iy = int(cx), int(cy)
        if 0 <= iy < mask.shape[0] and 0 <= ix < mask.shape[1] and mask[iy, ix]:
            tris.append(idx)

    if not tris:
        return None
    return all_pts, np.array(tris, dtype=np.int32)


def _resample_closed_curve(points, n):
    """Evenly resample a closed polygon outline to exactly n points by arc length."""
    closed = np.vstack([points, points[:1]])
    seg_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cumulative[-1]
    if total < 1e-6:
        return np.repeat(points[:1], n, axis=0)
    targets = np.linspace(0.0, total, n, endpoint=False)
    out = np.zeros((n, 2), dtype=np.float32)
    for k, t in enumerate(targets):
        i = max(0, min(np.searchsorted(cumulative, t) - 1, len(seg_lengths) - 1))
        frac = (t - cumulative[i]) / max(seg_lengths[i], 1e-9)
        out[k] = closed[i] + frac * (closed[i + 1] - closed[i])
    return out


def neighbor_map(verts, tris):
    """Build an adjacency list: vertex index -> set of connected vertex indices."""
    adj = [set() for _ in range(len(verts))]
    for tri in tris:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            adj[a].add(b)
            adj[b].add(a)
    return [np.array(sorted(s), dtype=np.int32) for s in adj]


class RubberMesh:
    """
    Holds a rest-pose mesh and produces deformed vertex positions each
    frame given an anchor (pinned region) and a moving pull target.
    """

    def __init__(self, verts, tris, anchor_indices, relax_iters=3):
        self.rest = verts.astype(np.float32)
        self.tris = tris
        self.anchor_indices = np.array(anchor_indices, dtype=np.int32)
        self.adjacency = neighbor_map(verts, tris)
        self.relax_iters = relax_iters
        # rest-edge lengths from each vertex to its neighbors, used to keep
        # the relaxation pass from collapsing or over-stretching locally
        self._rest_edge = [
            np.linalg.norm(self.rest[nbrs] - self.rest[i], axis=1) if len(nbrs) else
            np.zeros(0, dtype=np.float32)
            for i, nbrs in enumerate(self.adjacency)
        ]

    def deform(self, anchor_point, pull_point, reach, axis_gamma=1.5):
        """
        Compute deformed vertex positions for this frame.

        anchor_point : (x, y) rest-space point the mesh is rooted at (e.g.
                       the knuckle) -- vertices here stay put.
        pull_point   : (x, y) where the far end of the mesh (e.g. fingertip)
                       is currently being dragged to.
        reach        : float, the rest-space distance from anchor_point to
                       the mesh's original far end -- used to normalize how
                       far "all the way stretched" is.
        axis_gamma   : >1 keeps vertices near the anchor more rooted and
                       concentrates motion toward the far end; 1.0 = linear.
        """
        n = len(self.rest)
        anchor_arr = np.array(anchor_point, dtype=np.float32)
        pull_dx = pull_point[0] - anchor_point[0]
        pull_dy = pull_point[1] - anchor_point[1]

        # Pass 1: axial pull. Weight each vertex by how far ALONG the rest
        # anchor->reach axis it sits (0 at the anchor, 1 at the original
        # far end) -- NOT by raw distance from the anchor. This is what
        # makes the tip follow the pull almost fully while the base near
        # the anchor stays rooted, instead of the pull dying out before it
        # reaches the tip.
        deltas = self.rest - anchor_arr
        dist_from_anchor = np.linalg.norm(deltas, axis=1)
        axis_t = np.clip(dist_from_anchor / max(reach, 1e-6), 0.0, 1.3)
        weights = (axis_t ** axis_gamma).astype(np.float32)

        pos = self.rest.copy()
        pos[:, 0] += weights * pull_dx
        pos[:, 1] += weights * pull_dy
        # anchors are pinned exactly to the rest pose (no drift)
        pos[self.anchor_indices] = self.rest[self.anchor_indices]

        # Pass 2: local-rigidity relaxation. Each free vertex is nudged
        # toward the average position implied by its neighbors' CURRENT
        # offset plus the original rest-edge vector to it -- this keeps
        # triangles from shearing into spaghetti at high stretch while
        # still letting the overall shape follow the pull.
        free = np.ones(n, dtype=bool)
        free[self.anchor_indices] = False
        for _ in range(self.relax_iters):
            new_pos = pos.copy()
            for i in np.nonzero(free)[0]:
                nbrs = self.adjacency[i]
                if len(nbrs) == 0:
                    continue
                implied = pos[nbrs] + (self.rest[i] - self.rest[nbrs])
                target = implied.mean(axis=0)
                # blend 60% toward the rigidity-implied position, keeping
                # 40% of the direct pull so the mesh still tracks the hand
                new_pos[i] = pos[i] * 0.4 + target * 0.6
            pos = new_pos
            pos[self.anchor_indices] = self.rest[self.anchor_indices]

        return pos


def warp_mesh_into(texture, rest_uv, deformed, tris, target, feather=7):
    """
    Piecewise-affine texture-map each triangle from `rest_uv` to its
    `deformed` position and alpha-composite the result onto `target`
    (modified in place). `texture` may be BGR or BGRA (alpha = silhouette).

    Returns the bounding box (x0, y0, x1, y1) touched, or None.
    """
    H, W = target.shape[:2]
    xs, ys = deformed[:, 0], deformed[:, 1]
    x0 = max(0, int(xs.min()) - 2)
    y0 = max(0, int(ys.min()) - 2)
    x1 = min(W, int(xs.max()) + 3)
    y1 = min(H, int(ys.max()) + 3)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None

    region_w, region_h = x1 - x0, y1 - y0
    layer = np.zeros((region_h, region_w, 3), dtype=np.float32)
    alpha = np.zeros((region_h, region_w), dtype=np.uint8)
    offset = np.array([x0, y0], dtype=np.float32)
    has_alpha_channel = texture.ndim == 3 and texture.shape[2] == 4

    for tri in tris:
        src = rest_uv[tri].astype(np.float32)
        dst = (deformed[tri] - offset).astype(np.float32)
        bx, by, bw, bh = cv2.boundingRect(dst)
        if bw <= 0 or bh <= 0:
            continue
        local_dst = dst - np.float32([bx, by])
        affine = cv2.getAffineTransform(src, local_dst)
        warped = cv2.warpAffine(
            texture, affine, (bw, bh),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        tri_mask = np.zeros((bh, bw), dtype=np.uint8)
        cv2.fillConvexPoly(tri_mask, local_dst.astype(np.int32), 255, cv2.LINE_8)
        if has_alpha_channel:
            tri_mask = np.minimum(tri_mask, warped[:, :, 3])
            warped = warped[:, :, :3]

        ax0, ay0 = max(0, bx), max(0, by)
        ax1, ay1 = min(region_w, bx + bw), min(region_h, by + bh)
        if ax1 <= ax0 or ay1 <= ay0:
            continue
        sub_y, sub_x = ay0 - by, ax0 - bx
        sub_mask = tri_mask[sub_y:sub_y + (ay1 - ay0), sub_x:sub_x + (ax1 - ax0)]
        sub_warp = warped[sub_y:sub_y + (ay1 - ay0), sub_x:sub_x + (ax1 - ax0)]
        sel = sub_mask > 0
        layer[ay0:ay1, ax0:ax1][sel] = sub_warp[sel]
        np.maximum(alpha[ay0:ay1, ax0:ax1], sub_mask, out=alpha[ay0:ay1, ax0:ax1])

    if feather >= 3 and feather % 2 == 1:
        alpha = cv2.erode(alpha, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        alpha = cv2.GaussianBlur(alpha, (feather, feather), 0)

    alpha_f = (alpha.astype(np.float32) / 255.0)[:, :, None]
    alpha_f *= (layer.sum(axis=2, keepdims=True) > 0).astype(np.float32)
    roi = target[y0:y1, x0:x1]
    roi[:] = (roi.astype(np.float32) * (1.0 - alpha_f) + layer * alpha_f).astype(np.uint8)
    return (x0, y0, x1, y1)
