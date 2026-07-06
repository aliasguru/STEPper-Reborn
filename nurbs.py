import numpy as np
from mathutils import Vector


class NurbsPoint:
    def __init__(self, point):
        assert len(point) == 3 or len(point) == 4

        self.point = point
        self.x = point[0]
        self.y = point[1]
        self.z = point[2]
        if len(point) == 3:
            self.w = 1.0
        else:
            self.w = point[3]

    def location(self):
        return (self.x, self.y, self.z)

    def as_vector(self):
        return Vector((self.x, self.y, self.z, self.w))


class NurbsData:
    def __init__(self, uv_points):
        assert len(uv_points) > 1
        assert isinstance(uv_points[0][0], NurbsPoint)
        self.uv_points = uv_points

        self.u_closed = False
        self.v_closed = False

        self.u_periodic = False
        self.v_periodic = False

        self.u_degree = 2
        self.v_degree = 2


class NurbsCurveData:
    """Control-point representation of a single NURBS curve (1D)."""

    def __init__(self, points):
        assert len(points) >= 2
        assert isinstance(points[0], NurbsPoint)
        self.points = points

        self.degree = 3
        self.closed = False
        self.periodic = False

        # Unique knot values (ascending) and their multiplicities, parallel
        # arrays. Empty when the source curve didn't provide a knot vector
        # (e.g. built procedurally rather than read from a CAD file); degree
        # reduction is skipped in that case since it needs the real knots.
        self.knots = []
        self.multiplicities = []


# ---------------------------------------------------------------------------
# Degree reduction for curves above Blender's supported NURBS degree (5, i.e.
# order 6). Blender curves can only use a uniform or clamped-uniform knot
# vector (order_u + use_endpoint_u/use_cyclic_u) - there is no API to set an
# arbitrary knot vector. So a mathematically-reduced curve, which generally
# has a non-uniform knot vector, cannot be stored as a single Blender spline.
#
# Instead each curve is decomposed into Bezier segments (knot insertion up to
# full multiplicity at every interior knot), each segment is degree-reduced
# on its own, and the segments are emitted as separate, single-span Blender
# NURBS splines that are C0-continuous at their shared endpoints. A
# single-span clamped spline has no interior knots, so it fits Blender's
# uniform knot model exactly - this is what "more spans, lower degree" in the
# CAD comparison actually corresponds to.
#
# Degree reduction itself is done as the least-squares inverse of Bezier
# degree elevation (elevation has a simple, unambiguous closed-form matrix;
# reduction is its pseudo-inverse). For a curve that was originally elevated
# from a lower degree - the common case for higher-degree CAD export - this
# recovers the exact original control points (residual ~ 1e-14); for a
# genuinely higher-degree shape it gives the closest lower-degree
# approximation, with the fit error reported back to the caller.
# ---------------------------------------------------------------------------


def _to_homogeneous(points):
    """List[NurbsPoint] -> (N, 4) array of (w*x, w*y, w*z, w)."""
    return np.array([[p.x * p.w, p.y * p.w, p.z * p.w, p.w] for p in points], dtype=float)


def _from_homogeneous(arr):
    """(N, 4) homogeneous array -> List[NurbsPoint]."""
    pts = []
    for row in arr:
        w = row[3]
        pts.append(NurbsPoint((row[0] / w, row[1] / w, row[2] / w, w)))
    return pts


def _insert_knot_once(knot_vector, ctrl_pts, degree, u):
    """Boehm single-knot-insertion: raise the multiplicity of `u` by one.

    knot_vector: flat knot vector (list of floats, one entry per multiplicity).
    ctrl_pts: (N, 4) homogeneous control points.
    Returns (new_knot_vector, new_ctrl_pts); the curve shape is unchanged.
    """
    U = knot_vector
    p = degree
    n = len(ctrl_pts) - 1

    k = None
    for i in range(len(U) - 1):
        if U[i] <= u < U[i + 1]:
            k = i
    if k is None:
        raise ValueError("knot value out of range for insertion")

    s = sum(1 for x in U if abs(x - u) < 1e-9)

    new_ctrl = np.zeros((n + 2, 4))
    new_ctrl[: k - p + 1] = ctrl_pts[: k - p + 1]
    new_ctrl[k - s + 1 :] = ctrl_pts[k - s :]

    for i in range(k - p + 1, k - s + 1):
        alpha = (u - U[i]) / (U[i + p] - U[i])
        new_ctrl[i] = alpha * ctrl_pts[i] + (1 - alpha) * ctrl_pts[i - 1]

    new_U = U[: k + 1] + [u] + U[k + 1 :]
    return new_U, new_ctrl


def _decompose_to_bezier(knot_vector, ctrl_pts, degree):
    """Insert interior knots up to full multiplicity and split into Bezier
    segments. Returns a list of (degree + 1)-point homogeneous control point
    blocks, one per knot span, each sharing its boundary point with its
    neighbour.
    """
    U = list(knot_vector)
    Pw = ctrl_pts
    p = degree

    interior = sorted(set(round(u, 9) for u in U[p + 1 : -(p + 1)]))
    for u in interior:
        mult = sum(1 for x in U if abs(x - u) < 1e-9)
        for _ in range(p - mult):
            U, Pw = _insert_knot_once(U, Pw, p, u)

    segments = []
    i = 0
    while i + p < len(Pw):
        segments.append(Pw[i : i + p + 1])
        i += p
    return segments


def _elevation_matrix(p):
    """(p + 1, p) matrix elevating a Bezier curve of degree p-1 to degree p."""
    E = np.zeros((p + 1, p))
    for i in range(p + 1):
        if 0 <= i - 1 < p:
            E[i, i - 1] = i / p
        if i < p:
            E[i, i] += 1 - i / p
    return E


def _reduce_bezier_segment(ctrl_pts, degree):
    """Reduce one Bezier segment (degree+1 homogeneous control points) by one
    degree, fitting the interior control points by least squares against
    E @ Q = ctrl_pts while forcing both endpoints (Q_0, Q_last) to match the
    input exactly.

    Endpoints must be exact rather than least-squares-fitted: they are the
    points shared with neighbouring segments (and, at the ends of the curve,
    the points other geometry is attached to), so letting them drift would
    open visible gaps and move points that should stay fixed in space.

    Returns (reduced_ctrl_pts, max_abs_residual). The residual is ~0 when the
    segment is an exact elevation of a lower-degree curve.
    """
    p = degree
    E = _elevation_matrix(p)  # shape (p + 1, p)
    P0, Plast = ctrl_pts[0], ctrl_pts[-1]

    n_interior = p - 2
    if n_interior > 0:
        rhs = ctrl_pts - np.outer(E[:, 0], P0) - np.outer(E[:, -1], Plast)
        Q_mid, *_ = np.linalg.lstsq(E[:, 1:-1], rhs, rcond=None)
        Q = np.vstack([P0, Q_mid, Plast])
    else:
        Q = np.vstack([P0, Plast])[:p]

    residual = float(np.max(np.abs(E @ Q - ctrl_pts)))
    return Q, residual


def _split_bezier_segment(ctrl_pts, degree, t=0.5):
    """De Casteljau split of one Bezier segment into two, at local parameter
    t, via `degree` knot insertions. Both halves are exact (no approximation)
    and share the control point at t.
    """
    p = degree
    U = [0.0] * (p + 1) + [1.0] * (p + 1)
    Pw = ctrl_pts
    for _ in range(p):
        U, Pw = _insert_knot_once(U, Pw, p, t)
    return Pw[: p + 1], Pw[p:]


def _reduce_segment_adaptive(seg, source_degree, target_degree, tol, max_depth):
    """Reduce one Bezier segment from source_degree to target_degree.

    If the direct reduction chain's error exceeds `tol`, the segment is split
    in half (exact, no error) and each half is reduced recursively - this is
    what makes an inherently non-reducible high-degree curve (e.g. a genuine
    single-span degree 6 curve that isn't a plain elevation of a degree 5
    curve) converge to a good piecewise-lower-degree fit instead of forcing a
    single, badly-approximated low-degree segment. Recursion stops once the
    residual is within tolerance or max_depth is exhausted.

    Returns a list of (reduced_ctrl_pts, residual) leaves.
    """
    current = seg
    d = source_degree
    max_res = 0.0
    while d > target_degree:
        current, res = _reduce_bezier_segment(current, d)
        max_res = max(max_res, res)
        d -= 1

    if max_res <= tol or max_depth <= 0:
        return [(current, max_res)]

    left, right = _split_bezier_segment(seg, source_degree, 0.5)
    return _reduce_segment_adaptive(
        left, source_degree, target_degree, tol, max_depth - 1
    ) + _reduce_segment_adaptive(right, source_degree, target_degree, tol, max_depth - 1)


def reduce_curve_degree(cdata, max_degree=5, tol=None, max_subdivisions=8):
    """Reduce a NurbsCurveData above `max_degree` to a sequence of Bezier
    segments at `max_degree`, each ready to become its own single-span
    Blender NURBS spline.

    Existing (real) interior knots are decomposed first and always kept as
    segment boundaries. Each resulting segment is then reduced directly if
    that already meets `tol`; otherwise it is adaptively subdivided (see
    _reduce_segment_adaptive) until every piece does, up to max_subdivisions
    recursion levels. `tol` defaults to a small fraction of the curve's own
    bounding-box diagonal so it scales with the model rather than assuming a
    fixed unit system.

    Returns None if there is nothing to do (degree already <= max_degree) or
    reduction isn't possible (periodic curve, or no knot vector available).
    Otherwise returns (segments, max_residual, tol), where segments is a list
    of List[NurbsPoint] (each with max_degree + 1 points) and tol is the
    tolerance that was aimed for - callers should only treat max_residual as
    a problem if it exceeds tol (i.e. max_subdivisions was exhausted first).
    """
    if cdata.degree <= max_degree:
        return None
    if cdata.periodic or not cdata.knots:
        return None

    flat_knots = []
    for u, m in zip(cdata.knots, cdata.multiplicities):
        flat_knots.extend([u] * m)

    Pw = _to_homogeneous(cdata.points)
    degree = cdata.degree
    base_segments = _decompose_to_bezier(flat_knots, Pw, degree)

    if tol is None:
        bbox_diagonal = float(np.linalg.norm(np.ptp(Pw[:, :3] / Pw[:, 3:4], axis=0)))
        tol = 1e-4 * max(bbox_diagonal, 1e-6)

    all_segments = []
    max_residual = 0.0
    for seg in base_segments:
        for reduced, residual in _reduce_segment_adaptive(
            seg, degree, max_degree, tol, max_subdivisions
        ):
            max_residual = max(max_residual, residual)
            all_segments.append(reduced)

    return [_from_homogeneous(seg) for seg in all_segments], max_residual, tol
