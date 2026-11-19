"""numpy ports of the opencv functions vre_repository used to call. Same signatures and outputs (for the used flags)."""
# pylint: disable=invalid-name
import numpy as np

# same values as cv2's enums, so calls read (and compare) like the cv2 ones
cv2_RETR_CCOMP = 2
cv2_RETR_TREE = 3
cv2_CHAIN_APPROX_NONE = 1
cv2_CHAIN_APPROX_SIMPLE = 2

_CHAIN_DX = (1, 1, 0, -1, -1, -1, 0, 1) # chain codes: 0 = right, then counter-clockwise (y points down)
_CHAIN_DY = (0, -1, -1, -1, 0, 1, 1, 1)

def _follow_border(f: list[int], stride: int, i0: int, is_hole: bool, nbd: int, simple: bool) -> list[tuple[int, int]]:
    """Suzuki-Abe border following from flat index i0 (opencv's icvFetchContour). Marks f in place: -nbd where the
    right neighbour is background, else nbd (if not visited yet). Returns the (x, y) points in the padded image."""
    deltas = [dx + dy * stride for dx, dy in zip(_CHAIN_DX, _CHAIN_DY)]
    x, y = i0 % stride, i0 // stride
    s = s_end = 0 if is_hole else 4
    for _ in range(8): # clockwise from the background neighbour to the first foreground one
        s = (s - 1) & 7
        if f[i1 := i0 + deltas[s]] != 0 or s == s_end:
            break
    if s == s_end: # isolated pixel
        f[i0] = -nbd
        return [(x, y)]
    points, i3, prev_s = [], i0, s ^ 4
    for _ in range(8 * len(f) + 1): # every pixel is visited at most 4 times per border
        s_end = s
        for _ in range(8): # counter-clockwise from the previous pixel to the next foreground one
            s += 1
            if f[i4 := i3 + deltas[s & 7]] != 0:
                break
        s &= 7
        if 1 <= s <= s_end: # the right neighbour was examined and is background
            f[i3] = -nbd
        elif f[i3] == 1:
            f[i3] = nbd
        if not simple or s != prev_s:
            points.append((x, y))
        prev_s = s
        x, y = x + _CHAIN_DX[s], y + _CHAIN_DY[s]
        if i4 == i0 and i3 == i1:
            return points
        i3, s = i4, (s + 4) & 7
    raise RuntimeError("border following did not close")

def cv2_findContours(image: np.ndarray, mode: int, method: int) -> tuple[tuple[np.ndarray, ...], np.ndarray | None]:
    """
    cv2.findContours(image, mode, method) for mode RETR_CCOMP or RETR_TREE and method CHAIN_APPROX_NONE or _SIMPLE.
    Returns (contours, hierarchy) like opencv: contours are (N, 1, 2) int32 (x, y) arrays; hierarchy is (1, K, 4)
    int32 [next, previous, first_child, parent] (None if there are no contours). Any non-zero pixel is foreground.
    """
    assert mode in (cv2_RETR_CCOMP, cv2_RETR_TREE), mode
    assert method in (cv2_CHAIN_APPROX_NONE, cv2_CHAIN_APPROX_SIMPLE), method
    assert image.ndim == 2, image.shape
    h, w = image.shape
    padded = np.zeros((h + 2, w + 2), dtype=np.int64)
    padded[1:-1, 1:-1] = image != 0
    stride = w + 2
    f = padded.ravel().tolist()
    # raster scan over value changes only (opencv's icvFindNextContour): a border can only start where a run of
    # foreground starts (outer border) or ends (hole border)
    changes = np.flatnonzero(padded[:, 1:] != padded[:, :-1]) # (row, col) -> pixel col + 1 differs from col
    rows, cols = changes // (w + 1), changes % (w + 1) + 1
    row_bounds = np.searchsorted(rows, np.arange(h + 3)).tolist()
    contours, parents, is_holes, border_of_nbd = [], [], [], {}
    for y in range(1, h + 1):
        x_resume = 1
        for x in cols[row_bounds[y]:row_bounds[y + 1]].tolist():
            if x < x_resume: # we resume right after a border start
                continue
            i = y * stride + x
            prev, p = f[i - 1], f[i]
            is_hole = not (prev == 0 and p == 1)
            if is_hole and (p != 0 or prev < 1):
                continue
            lnbd = i - 1 # the last border pixel to the left on this row (row start = the frame)
            while lnbd > y * stride and f[lnbd] in (0, 1):
                lnbd -= 1
            parent = -1
            if lnbd > y * stride and (is_hole or mode == cv2_RETR_TREE):
                parent = border_of_nbd[abs(f[lnbd])]
                if is_holes[parent] == is_hole:
                    parent = parents[parent]
            nbd = len(contours) + 2
            border_of_nbd[nbd] = len(contours)
            contours.append(_follow_border(f, stride, i - is_hole, is_hole, nbd, method == cv2_CHAIN_APPROX_SIMPLE))
            parents.append(parent)
            is_holes.append(is_hole)
            x_resume = x + 1
    if len(contours) == 0:
        return (), None
    # opencv inserts each contour as the first child of its parent, then lists the tree depth-first
    children = {k: [] for k in range(-1, len(contours))}
    for k in reversed(range(len(contours))):
        children[parents[k]].append(k)
    order, stack = [], children[-1][::-1]
    while stack:
        order.append(k := stack.pop())
        stack.extend(children[k][::-1])
    position = {k: ix for ix, k in enumerate(order)}
    hierarchy = np.full((len(order), 4), -1, dtype=np.int32)
    for k in order:
        siblings = children[parents[k]]
        sib_ix = siblings.index(k)
        hierarchy[position[k]] = [position[siblings[sib_ix + 1]] if sib_ix + 1 < len(siblings) else -1,
                                  position[siblings[sib_ix - 1]] if sib_ix > 0 else -1,
                                  position[children[k][0]] if children[k] else -1,
                                  position[parents[k]] if parents[k] != -1 else -1]
    out = tuple(np.asarray(contours[k], dtype=np.int32).reshape(-1, 1, 2) - 1 for k in order) # unpad
    return out, hierarchy[None]

def _foreground_runs(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """the runs of non-zero pixels of each row, in raster order: (row, x_start, x_end exclusive)"""
    fg = np.zeros((image.shape[0], image.shape[1] + 2), dtype=np.int8)
    fg[:, 1:-1] = image != 0
    edges = np.diff(fg, axis=1)
    run_y, run_x0 = np.nonzero(edges == 1)
    return run_y, run_x0, np.nonzero(edges == -1)[1]

def _connect_runs(run_y: np.ndarray, run_x0: np.ndarray, run_x1: np.ndarray, h: int, reach: int) -> np.ndarray:
    """union-find over runs: a run joins the runs of the row above that overlap it (reach=1: diagonals too).
    Returns each run's root, the smallest run index of its component."""
    root = list(range(len(run_y)))
    def find(a: int) -> int:
        while root[a] != a:
            root[a] = root[root[a]]
            a = root[a]
        return a
    row_start = np.searchsorted(run_y, np.arange(h + 1))
    for y in range(1, h):
        cur, above = slice(row_start[y], row_start[y + 1]), slice(row_start[y - 1], row_start[y])
        # the above runs that overlap run r are [first ending after r's start, first starting after r's end)
        lo = np.searchsorted(run_x1[above], run_x0[cur] - reach, side="right") + above.start
        hi = np.searchsorted(run_x0[above], run_x1[cur] + reach, side="left") + above.start
        for r, a0, a1 in zip(range(cur.start, cur.stop), lo.tolist(), hi.tolist()):
            for a in range(a0, a1):
                ra, rr = find(a), find(r)
                root[max(ra, rr)] = min(ra, rr)
    return np.array([find(r) for r in range(len(run_y))], dtype=np.int64)

def cv2_connectedComponentsWithStats(image: np.ndarray, connectivity: int = 8) \
        -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    cv2.connectedComponentsWithStats(image, connectivity) with opencv's label order. Returns (n_labels, labels (H, W)
    int32, stats (n_labels, 5) int32 [left, top, width, height, area], centroids (n_labels, 2) float64 [x, y]).
    Label 0 is the background (zero pixels).
    """
    assert connectivity in (4, 8), connectivity
    assert image.ndim == 2, image.shape
    h, w = image.shape
    run_y, run_x0, run_x1 = _foreground_runs(image)
    n_runs = len(run_y)
    run_root = _connect_runs(run_y, run_x0, run_x1, h, reach=1 if connectivity == 8 else 0)
    # opencv scans 2x2 blocks (Spaghetti for 8-conn) or pixels (SAUF for 4-conn) and numbers components in the order
    # their first block/pixel is met. Order the components by that key.
    if connectivity == 8:
        key = (run_y // 2) * ((w + 1) // 2) + run_x0 // 2
    else:
        key = run_y * w + run_x0
    roots = np.unique(run_root)
    first_key = np.full(n_runs, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(first_key, run_root, key)
    label_of_root = np.zeros(n_runs, dtype=np.int32)
    label_of_root[roots[np.argsort(first_key[roots], kind="stable")]] = np.arange(1, len(roots) + 1, dtype=np.int32)
    run_label = label_of_root[run_root]
    n_labels = len(roots) + 1

    labels = np.zeros((h, w), dtype=np.int32)
    run_len = run_x1 - run_x0
    if n_runs > 0:
        offset_in_run = np.arange(run_len.sum()) - np.repeat(np.cumsum(run_len) - run_len, run_len)
        flat = np.repeat(run_y * w + run_x0, run_len) + offset_in_run
        labels.ravel()[flat] = np.repeat(run_label, run_len)

    stats = np.zeros((n_labels, 5), dtype=np.int32)
    centroids = np.zeros((n_labels, 2), dtype=np.float64)
    area = np.bincount(run_label, weights=run_len, minlength=n_labels)[1:].astype(np.int64)
    left = np.full(n_labels - 1, w, dtype=np.int64)
    np.minimum.at(left, run_label - 1, run_x0)
    right = np.full(n_labels - 1, -1, dtype=np.int64)
    np.maximum.at(right, run_label - 1, run_x1 - 1)
    top = np.full(n_labels - 1, h, dtype=np.int64)
    np.minimum.at(top, run_label - 1, run_y)
    bottom = np.full(n_labels - 1, -1, dtype=np.int64)
    np.maximum.at(bottom, run_label - 1, run_y)
    sum_x = np.bincount(run_label, weights=(run_x0 + run_x1 - 1) * run_len / 2, minlength=n_labels)[1:]
    sum_y = np.bincount(run_label, weights=run_y * run_len, minlength=n_labels)[1:]
    stats[1:] = np.stack([left, top, right - left + 1, bottom - top + 1, area], axis=1)
    centroids[1:] = np.stack([sum_x / np.maximum(area, 1), sum_y / np.maximum(area, 1)], axis=1)

    bg_y, bg_x = np.nonzero(labels == 0) # background: bounding box and centroid of the zero pixels
    if len(bg_y) > 0:
        stats[0] = [bg_x.min(), bg_y.min(), bg_x.max() - bg_x.min() + 1, bg_y.max() - bg_y.min() + 1, len(bg_y)]
        centroids[0] = [bg_x.sum() / len(bg_x), bg_y.sum() / len(bg_y)]
    else: # no zero pixels: opencv's untouched accumulators
        stats[0] = [-1, np.iinfo(np.int32).max, 0, 0, 0]
        centroids[0] = np.nan
    return n_labels, labels, stats, centroids
