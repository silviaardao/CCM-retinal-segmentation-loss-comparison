"""Derive crossover and branch-region masks for the CBAV-Loss adaptation.

References
----------
CBAV-Loss:
Zhang et al. (2023). DOI: 10.1007/978-981-99-8558-6_5.

Skeletonisation:
Zhang and Suen (1984), "A fast parallel algorithm for thinning digital
patterns". DOI: 10.1145/357994.358023.

The artery and vein ground-truth masks are thinned to one-pixel-wide
skeletons, SA and SV. Structural masks are then defined as:

    Mcross   = SA & SV
    MbranchA = artery-skeleton pixels with at least three neighbours
    MbranchV = vein-skeleton pixels with at least three neighbours

The five-class fundus annotations store artery-vein crossings as a separate
overlap class (class 3). To preserve that structure, overlap pixels are
included in both the artery and vein masks before skeletonisation. Otherwise,
the separately encoded overlap pixels would not appear in the intersection
of the two skeletons.

The masks are derived from the ground-truth labels using NumPy and are treated
as fixed spatial masks. Gradients are calculated for the loss applied at these
locations, but not through the mask-generation procedure itself.

This is a study-specific adaptation of CBAV-Loss to the five-class fundus
annotation scheme.
"""

import numpy as np


def _shift(arr, dy, dx):
    """Shift a boolean array by (dy, dx) with zero padding."""
    h, w = arr.shape
    out = np.zeros_like(arr, dtype=bool)
    ys0, ys1 = max(0, -dy), min(h, h - dy)
    xs0, xs1 = max(0, -dx), min(w, w - dx)
    yd0, yd1 = max(0, dy), min(h, h + dy)
    xd0, xd1 = max(0, dx), min(w, w + dx)
    out[yd0:yd1, xd0:xd1] = arr[ys0:ys1, xs0:xs1]
    return out


def _neighbour_count(arr):
    count = np.zeros(arr.shape, dtype=np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            count += _shift(arr, dy, dx).astype(np.uint8)
    return count


def zhang_suen_skeletonize(binary, max_iter=50):
    """Pure-NumPy Zhang-Suen thinning (DOI: 10.1145/357994.358023)."""
    img = binary.astype(bool).copy()
    for _ in range(max_iter):
        before = img.copy()
        for step in (0, 1):
            p2 = _shift(img, -1, 0); p3 = _shift(img, -1, 1)
            p4 = _shift(img, 0, 1);  p5 = _shift(img, 1, 1)
            p6 = _shift(img, 1, 0);  p7 = _shift(img, 1, -1)
            p8 = _shift(img, 0, -1); p9 = _shift(img, -1, -1)
            nb = [p2, p3, p4, p5, p6, p7, p8, p9]
            n = sum(x.astype(np.uint8) for x in nb)
            trans = sum((~nb[i] & nb[(i + 1) % 8]).astype(np.uint8) for i in range(8))
            if step == 0:
                m = img & (n >= 2) & (n <= 6) & (trans == 1) & ~(p2 & p4 & p6) & ~(p4 & p6 & p8)
            else:
                m = img & (n >= 2) & (n <= 6) & (trans == 1) & ~(p2 & p4 & p8) & ~(p2 & p6 & p8)
            img[m] = False
        if np.array_equal(img, before):
            break
    return img


def cbav_point_masks(label, artery_ids=(1, 3), vein_ids=(2, 3)):
    """
    Build the CBAV crossover/branch point masks from a 2D class-index label.

    Returns bool arrays (Mcross, MbranchA, MbranchV). Overlap (class 3) counts as
    both artery and vein so skeletons pass through crossings and intersect there.
    """
    artery = np.isin(label, artery_ids)
    vein = np.isin(label, vein_ids)

    sa = zhang_suen_skeletonize(artery)
    sv = zhang_suen_skeletonize(vein)

    m_cross = sa & sv
    m_branch_a = sa & (_neighbour_count(sa) >= 3)
    m_branch_v = sv & (_neighbour_count(sv) >= 3)
    return m_cross, m_branch_a, m_branch_v
