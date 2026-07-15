import logging
import ntpath
import time
import numpy as np
import bmesh
import bpy
from collections import defaultdict
from . import nurbs
from .trimesh import TriMesh
from .utils import (
    set_obj_matrix_world,
    obj_unlink_all,
    transform_to_up,
    create_new_obj_with_mesh,
    faces_of_edges,
    vert_coordinates_of_edges,
    vert_of_edges,
    bpy_update_object_data,
    choose_hierarchy_types,
    GetAddonPreferences,
    freeze_matrix,
)

logger = logging.getLogger(__package__)

GLOBAL_FILE_CACHE = {}

# Blender NURBS splines support order_u 2..6, i.e. degree 1..5.
BLENDER_MAX_CURVE_DEGREE = 5


def project_and_compare_normals(proj_plane_normal, norms_a, norms_b, margin_sq):
    """
    Normals are projected and compared.
    Vectorized.
    Computation is simplified so it doesn't resemble a direct projection with normalized vectors.
    """
    dot_ab = np.einsum("ij,ij->i", norms_a, norms_b)
    ea = np.einsum("ij,ij->i", proj_plane_normal, norms_a)
    eb = np.einsum("ij,ij->i", proj_plane_normal, norms_b)

    proj_dot = dot_ab - ea * eb
    len_sq_a = 1.0 - ea * ea
    len_sq_b = 1.0 - eb * eb

    # not sharp if either normal is too small
    are_too_small = np.logical_or(np.less(len_sq_a, 1e-12), np.less(len_sq_b, 1e-12))

    over_margin = np.less((proj_dot * proj_dot), margin_sq * len_sq_a * len_sq_b)
    return np.logical_and(over_margin, np.logical_not(are_too_small))


def find_seams(trimesh, face_pair_of_edges):
    """
    Mark boundaries between different shape batches
    """
    # TODO: make batches a mono-block array of TriMesh
    batches = [t.batch for t in trimesh.tris]
    is_seam = np.bool([batches[f1] != batches[f2] for f1, f2 in face_pair_of_edges])
    return is_seam


def find_sharp(ob_data, trimesh, face_pair_of_edges, margin):
    # Sharp edges: mark where per-vertex normals are discontinuous
    edge_count = len(ob_data.edges)

    # for each tris, stores all normals of the 3 vertices
    face_vert_norms = [
        {
            face.indices[0]: face.norms[0],
            face.indices[1]: face.norms[1],
            face.indices[2]: face.norms[2],
        }
        for face in trimesh.tris
    ]

    # get the 3 verts normals for the 2 faces of each edges
    norms_face1 = [face_vert_norms[f1] for f1, _ in face_pair_of_edges]
    norms_face2 = [face_vert_norms[f2] for _, f2 in face_pair_of_edges]

    # Vert ids of each edge
    edge_verts = vert_of_edges(ob_data)

    # Get normals for each face of each vertex of each edge
    norm_face1_vert1 = np.empty((edge_count, 3), dtype=np.float32)
    norm_face2_vert1 = np.empty((edge_count, 3), dtype=np.float32)
    norm_face1_vert2 = np.empty((edge_count, 3), dtype=np.float32)
    norm_face2_vert2 = np.empty((edge_count, 3), dtype=np.float32)

    get_fallback = lambda a: (a if a is not None else np.zeros(3, dtype=np.float32))
    for i in range(edge_count):
        v1, v2 = edge_verts[i]
        norm_face1_vert1[i] = get_fallback(norms_face1[i].get(v1))
        norm_face2_vert1[i] = get_fallback(norms_face2[i].get(v1))
        norm_face1_vert2[i] = get_fallback(norms_face1[i].get(v2))
        norm_face2_vert2[i] = get_fallback(norms_face2[i].get(v2))

    # Edge normal plane's normal vector
    vert_co_0, vert_co_1 = vert_coordinates_of_edges(ob_data, edge_verts)
    edge_dir = vert_co_0 - vert_co_1

    # margin squared is used for faster computation
    margin_sq = (1 - margin) ** 2
    is_vert1_sharp = project_and_compare_normals(
        edge_dir, norm_face1_vert1, norm_face2_vert1, margin_sq
    )
    is_vert2_sharp = project_and_compare_normals(
        edge_dir, norm_face1_vert2, norm_face2_vert2, margin_sq
    )
    is_sharp = np.logical_and(is_vert1_sharp, is_vert2_sharp)

    return is_sharp


def mark_edges(objdata, trimesh: TriMesh, seams, sharp, margin=0.02):
    if not (seams or sharp):
        return

    # Face pair of edges
    foe = faces_of_edges(objdata)
    face_pair_of_edges = np.zeros((len(objdata.edges), 2), dtype=np.int32)
    for i in range(len(face_pair_of_edges)):
        foei = foe[i]
        if len(foei) == 2:
            face_pair_of_edges[i, 0] = foei[0]
            face_pair_of_edges[i, 1] = foei[1]

    # Compute and createattributes
    if seams:
        is_seam = find_seams(trimesh, face_pair_of_edges)
        if "uv_seam" not in objdata.attributes:
            objdata.attributes.new(name="uv_seam", type="BOOLEAN", domain="EDGE")
    if sharp:
        is_sharp = find_sharp(objdata, trimesh, face_pair_of_edges, margin)
        if "sharp_edge" not in objdata.attributes:
            objdata.attributes.new(name="sharp_edge", type="BOOLEAN", domain="EDGE")
    objdata.update()

    # Get and Set attributes
    if seams:
        seam_att = objdata.attributes["uv_seam"]
        seam_att.data.foreach_set("value", is_seam)
    if sharp:
        sharp_att = objdata.attributes["sharp_edge"]
        sharp_att.data.foreach_set("value", is_sharp)


def build_mesh(
    context,
    step_reader,
    obj,
    shp,
    lind,
    angd,
    vcol_name="Colors",
    edges_as_seams=True,
    discontinuity_as_sharp=True,
):
    prefs = GetAddonPreferences(context)
    hacks = set([])
    if prefs.hack_skip_zero_solids:
        hacks.add("skip_solids")

    import time

    start_time = time.time()

    trimesh: TriMesh = step_reader.build_trimesh(
        shp, lin_def=lind, ang_def=angd, hacks=hacks
    )

    end_time = time.time()
    print(f"Trimesh build time: {end_time - start_time:.2f} seconds")

    trimesh.fuse_verts()
    trimesh.filter_zero_area()
    trimesh.filter_same_face()
    trimesh.fill_empty_color()
    # Clear any existing geometry first: from_pydata() only works on an empty
    # mesh, so a rebuild (mesh already populated) would otherwise raise
    # "internal error setting the array".  No-op on a freshly created mesh.
    obj.data.clear_geometry()
    trimesh.add_to_mesh(obj.data)
    obj.data.update()

    if trimesh.tris:
        mark_edges(
            obj.data, trimesh, edges_as_seams, discontinuity_as_sharp, margin=0.02
        )

    bpy_update_object_data(
        obj.data,
        vcol_name,
        trimesh.get_loop_colors(),
        trimesh.get_loop_uvs(),
        trimesh.get_loop_normals(),
        trimesh.get_loop_material_names(),
        build_materials=prefs.build_materials,
    )

    return trimesh.matrix


def build_nurbs(step_reader, shp, name):
    nurbs_data = step_reader.build_nurbs(shp)
    debug_faces = False
    if debug_faces:
        obj = create_new_obj_with_mesh(name)
        bm = bmesh.new()
        for nb in nurbs_data:
            nb_u = nb.uv_points
            uw, vw = len(nb_u), len(nb_u[0])
            for u in range(uw - 1):
                nb_v0 = nb_u[u]
                nb_v1 = nb_u[u + 1]
                for v in range(vw - 1):
                    a = bm.verts.new(nb_v0[v].location())
                    b = bm.verts.new(nb_v0[v + 1].location())
                    c = bm.verts.new(nb_v1[v + 1].location())
                    d = bm.verts.new(nb_v1[v].location())
                    bm.faces.new((d, c, b, a))
        prev_mode = bpy.context.object.mode
        bm.to_mesh(obj.data)
        # obj.display_type = 'WIRE'
        return obj
    else:
        blender_nurbs = []
        for nb in nurbs_data:
            surface_data = bpy.data.curves.new("wook", "SURFACE")
            surface_data.dimensions = "3D"

            upoints = nb.uv_points

            usize, vsize = len(upoints), len(upoints[0])

            splines = []
            for v in range(usize):
                spline = surface_data.splines.new(type="NURBS")
                spline.points.add(vsize - 1)
                splines.append(spline)

            for ui, vpoints in enumerate(upoints):
                for vi, p in enumerate(vpoints):
                    # points have weight attribute
                    splines[ui].points[vi].co = p.as_vector()

            blender_nurbs.append(surface_data)

        # print(dir(nurbs[0].splines[0])) =>
        # 'bezier_points', 'bl_rna', 'calc_length', 'character_index', 'hide', 'material_index',
        # 'order_u', 'order_v', 'point_count_u', 'point_count_v', 'points', 'radius_interpolation',
        # 'resolution_u', 'resolution_v', 'rna_type', 'tilt_interpolation', 'type', 'use_bezier_u',
        # 'use_bezier_v', 'use_cyclic_u', 'use_cyclic_v', 'use_endpoint_u',
        # 'use_endpoint_v', 'use_smooth'
        created_objs = []
        for ni, n in enumerate(blender_nurbs):
            occ_nurb = nurbs_data[ni]
            surface_object = bpy.data.objects.new(name, n)
            bpy.context.collection.objects.link(surface_object)
            for s in surface_object.data.splines:
                for p in s.points:
                    p.select = True

            bpy.context.view_layer.objects.active = surface_object
            prev_mode = bpy.context.object.mode
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.curve.make_segment()
            bpy.ops.object.mode_set(mode=prev_mode)
            created_objs.append(surface_object)

        for obi, ob in enumerate(created_objs):
            occ_nurb = nurbs_data[obi]
            for s in ob.data.splines:
                s.use_endpoint_u = True
                s.use_endpoint_v = True
                # s.use_endpoint_u = occ_nurb.u_closed
                # s.use_endpoint_v = occ_nurb.v_closed
                # s.use_cyclic_u = occ_nurb.u_periodic
                # s.use_cyclic_v = occ_nurb.v_periodic
                s.order_u = occ_nurb.u_degree + 1
                s.order_v = occ_nurb.v_degree + 1
                # print(s.order_u, s.order_v, occ_nurb.u_degree, occ_nurb.v_degree)

        # Join objects
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        for o in created_objs:
            o.select_set(True)
        bpy.ops.object.join()
        return bpy.context.view_layer.objects.active


def _add_nurbs_spline(curve_data, pts, degree, periodic):
    spline = curve_data.splines.new("NURBS")
    spline.points.add(len(pts) - 1)  # one point exists by default
    for i, p in enumerate(pts):
        spline.points[i].co = p.as_vector()

    # Blender requires 2 <= order_u <= 6 and order_u <= point count.
    spline.order_u = max(2, min(degree + 1, len(pts), 6))
    # Higher display/evaluation resolution than the default (12) for smoother
    # curves; 30 is a good trade-off for typical CAD curves.
    spline.resolution_u = 30
    if periodic:
        spline.use_cyclic_u = True
    else:
        spline.use_endpoint_u = True


def build_curves_object(curve_data_list, name):
    """Build a single CURVE object from parsed NURBS curves.

    Note: Blender NURBS splines cannot store arbitrary knot vectors. Clamped and
    periodic ends are set via use_endpoint_u / use_cyclic_u, which matches the
    common (clamped) CAD b-splines well; strongly non-uniform interior knots may
    deviate slightly from the original geometry.

    Curves with a degree above Blender's supported maximum (5) are degree-reduced
    via nurbs.reduce_curve_degree and emitted as multiple single-span splines -
    one per degree-reduced Bezier segment - since Blender can only represent a
    single-span clamped spline exactly. This is why such a curve turns into more
    (lower-degree) spans, matching how CAD packages perform the same conversion.
    Curves that can't be reduced (periodic, or missing a source knot vector) fall
    back to the previous behaviour of clamping order_u, which may deviate in shape.
    """
    curve_data = bpy.data.curves.new(name, "CURVE")
    curve_data.dimensions = "3D"

    for cdata in curve_data_list:
        reduced = None
        if cdata.degree > BLENDER_MAX_CURVE_DEGREE:
            try:
                reduced = nurbs.reduce_curve_degree(cdata, BLENDER_MAX_CURVE_DEGREE)
            except Exception:
                logger.warning(
                    "Curve %r: degree reduction from %d to %d failed, "
                    "falling back to order clamping",
                    name, cdata.degree, BLENDER_MAX_CURVE_DEGREE, exc_info=True,
                )

        if reduced is not None:
            segments, max_residual, tol = reduced
            if max_residual > tol:
                logger.warning(
                    "Curve %r: degree %d -> %d reduction could not reach the "
                    "target tolerance (max control point error %.4g > %.4g), "
                    "shape may deviate slightly",
                    name, cdata.degree, BLENDER_MAX_CURVE_DEGREE, max_residual, tol,
                )
            for seg_pts in segments:
                _add_nurbs_spline(curve_data, seg_pts, BLENDER_MAX_CURVE_DEGREE, periodic=False)
        else:
            _add_nurbs_spline(curve_data, cdata.points, cdata.degree, cdata.periodic)

    obj = bpy.data.objects.new(name, curve_data)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    return obj


def load_step(
    context,
    filepath,
    custom_scale=None,
    lin_deflection=0.8,
    ang_deflection=0.5,
    # merge_distance=0.001,
    up_as="Y",
    htypes="TREE",
):
    from . import importer

    hierarchy_flat, hierarchy_tree, hierarchy_empties = choose_hierarchy_types(htypes)

    filename = "".join(ntpath.basename(filepath).split(".")[:-1])

    if filepath not in GLOBAL_FILE_CACHE:
        try:
            step_reader = importer.ReadSTEP(filepath)
            GLOBAL_FILE_CACHE[filepath] = step_reader
        except AssertionError as e:
            print(e)
            return False

    else:
        step_reader = GLOBAL_FILE_CACHE[filepath]
        print("Loaded file from cache")

    prefs = GetAddonPreferences(context)

    tree = step_reader.tree
    scale = step_reader.scale
    if custom_scale is not None:
        scale = custom_scale

    # divide by Blender unit length
    scale /= context.scene.unit_settings.scale_length
    print("Current Blender scale set at:", context.scene.unit_settings.scale_length)

    wm = bpy.context.window_manager

    created_objs = []
    created_names = {}
    created_curve_names = {}
    created_uuid = {}

    # traverse shapes, render in "face" mode
    start_time = time.time()
    all_shapes = tree.get_shapes()
    total = len(all_shapes)

    # Build mesh objects
    wm.progress_begin(0, total)
    for i, (shp, node_index) in enumerate(all_shapes):
        parent_uuid, self_uuid, tag, obj_name, _, local_t, global_t = tree.nodes[
            node_index
        ].get_values()

        if obj_name == "root":
            obj_name = filename + ".empties"

        shape_name = "tt_" + repr(tag)
        wm.progress_update(i)
        obj = None

        # Shape found in leaf
        if shp:
            print(
                "\nBuilding ({}/{}): {} ".format(i + 1, total, obj_name),
                end="",
                flush=True,
            )
            print("[T" + repr(shp.ShapeType()) + "]", end="", flush=True)

            # If object already build, just copy it, using linked mesh data
            if shape_name in created_names:
                print("[Link]", end="", flush=True)

                source_obj = created_names[shape_name]
                obj = source_obj.copy()
                created_objs.append(obj)

                # Instance the curve sibling too, if this shape produced one
                if shape_name in created_curve_names:
                    curve_obj = created_curve_names[shape_name].copy()
                    curve_obj["STEP_tag"] = tag
                    curve_obj["STEP_parent"] = parent_uuid
                    curve_obj["STEP_uuid"] = self_uuid
                    curve_obj["STEP_file"] = filepath
                    curve_obj["STEP_name"] = obj_name
                    curve_obj["STEP_tree_location"] = node_index
                    created_objs.append(curve_obj)
            else:
                print("[Build]", end="", flush=True)

                # Optional curve import: extract free-standing edges/wires.
                curve_data_list = (
                    step_reader.build_curves(shp) if prefs.import_curves else []
                )
                # A shape with no faces is a pure wireframe; building a mesh for
                # it would only create an empty Mesh container. Such shapes become
                # CURVE objects directly when curve import yields geometry.
                build_as_curve_only = (
                    bool(curve_data_list) and not step_reader.has_faces(shp)
                )

                if build_as_curve_only:
                    obj = build_curves_object(curve_data_list, obj_name)
                    created_objs.append(obj)
                    created_names[shape_name] = obj
                else:
                    # Create new mesh and object from scratch
                    obj = create_new_obj_with_mesh(obj_name)
                    bpy.ops.object.mode_set(mode="OBJECT")
                    build_mesh(
                        context, step_reader, obj, shp, lin_deflection, ang_deflection
                    )

                    # TODO: nurbs changes here
                    # obj = build_nurbs(step_reader, shp, name)

                    created_objs.append(obj)
                    created_names[shape_name] = obj

                    # Mixed shape (faces + free curves): the curve object is a
                    # sibling sharing the same tree node, so hierarchy and
                    # transforms apply identically.
                    if curve_data_list:
                        curve_obj = build_curves_object(
                            curve_data_list, obj_name + ".curves"
                        )
                        curve_obj["STEP_tag"] = tag
                        curve_obj["STEP_parent"] = parent_uuid
                        curve_obj["STEP_uuid"] = self_uuid
                        curve_obj["STEP_file"] = filepath
                        curve_obj["STEP_name"] = obj_name
                        curve_obj["STEP_tree_location"] = node_index
                        created_objs.append(curve_obj)
                        created_curve_names[shape_name] = curve_obj

                # bpy.ops.object.mode_set(mode="OBJECT")
                # build_mesh(step_reader, obj, shp, lin_deflection, ang_deflection)

        # No shape in leaf, empty creation enabled, do this
        elif hierarchy_empties:
            # Create empty
            obj = bpy.data.objects.new(obj_name, None)
            obj.empty_display_size = 2
            obj.empty_display_type = "PLAIN_AXES"
            created_objs.append(obj)
            # set_obj_matrix_world(obj, global_t)

        # Object has been created
        if obj:
            # assign property to obj
            obj["STEP_tag"] = tag
            obj["STEP_parent"] = parent_uuid
            obj["STEP_uuid"] = self_uuid
            obj["STEP_file"] = filepath
            obj["STEP_name"] = obj_name
            obj["STEP_tree_location"] = node_index
            created_uuid[self_uuid] = obj

    # assert len(created_objs) == len(shapes_labels)
    print("\n" + repr(step_reader.import_problems))

    # Store scale and up axis on each object so rebuild operations can
    # re-apply the transform later.
    for obj in created_objs:
        obj["STEP_scale"] = scale
        obj["STEP_up"] = up_as[0]

    # remove all temporary links
    for tobj in created_objs:
        obj_unlink_all(tobj)

    # build flat collection (one collection, all objects linked directly)
    if hierarchy_flat:
        flat_collection = bpy.data.collections.new(filename)
        bpy.context.scene.collection.children.link(flat_collection)

        for obj in created_objs:
            global_t = tree.nodes[obj["STEP_tree_location"]].global_transform
            set_obj_matrix_world(obj, global_t)
            flat_collection.objects.link(obj)

    # build tree of collections
    if hierarchy_tree:
        tree_collection = bpy.data.collections.new(filename + ".hierarchy")
        bpy.context.scene.collection.children.link(tree_collection)
        hierarchy_collections = {}
        # Objects whose immediate parent is the root node itself use the
        # root's own index (0) as STEP_parent; -1 only ever occurs as the
        # root node's own `parent` value. Map both to the top collection.
        hierarchy_collections[-1] = tree_collection
        hierarchy_collections[tree.get_root_id()] = tree_collection

        def node_parse(node, level, parent_collection):
            # if "name" in node and node["children"] is not None:
            if len(node.children) > 0:
                collection_node = bpy.data.collections.new(node.name)
                assert node.index not in hierarchy_collections
                hierarchy_collections[node.index] = collection_node

                parent_collection.children.link(collection_node)
                for c in node.children:
                    node_parse(tree.nodes[c], level + 1, collection_node)

        root = tree.nodes[0]
        if len(root.children) > 0:
            for c in root.children:
                node_parse(tree.nodes[c], 0, tree_collection)

            # link objects to tree
            if len(hierarchy_collections.items()) > 0:
                for obj in created_objs:
                    hierarchy_collections[obj["STEP_parent"]].objects.link(obj)
                    global_t = tree.nodes[obj["STEP_tree_location"]].global_transform
                    set_obj_matrix_world(obj, global_t)

    # build hierarchy with empties
    if hierarchy_empties:
        for obj in created_objs:
            global_t = tree.nodes[obj["STEP_tree_location"]].global_transform
            set_obj_matrix_world(obj, global_t)
            bpy.context.scene.collection.objects.link(obj)

            # Parent objs
            parent_id = obj["STEP_parent"]
            if parent_id in created_uuid:
                parent = created_uuid[parent_id]
                obj.parent = parent
                obj.matrix_parent_inverse = parent.matrix_world.inverted()

    transform_to_up(up_as[0], created_objs, scale)
    freeze_matrix(created_objs)

    wm.progress_end()
    print(f"STEP loading time elapsed: {time.time()-start_time:.2f}")

    return True
