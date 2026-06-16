# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Mesh building utilities: geometry processing, material creation, timing."""

import numpy as np
import bpy
from mathutils import Vector, Matrix
from OCP.AIS import AIS_Shape

# ---------------------------------------------------------------------------
# Color / material helpers
# ---------------------------------------------------------------------------

# Color quantization precision for material merging (~1.5% tolerance)
_COLOR_MERGE_PRECISION = 64


def _quantize_color(col):
    """Round color components to merge near-identical materials."""
    return tuple(
        round(c * _COLOR_MERGE_PRECISION) / _COLOR_MERGE_PRECISION for c in col
    )


def add_material(name, color, link_vertex_color=False, overwrite=False):
    assert len(color) == 3
    assert isinstance(color, tuple)
    if len(name) > 60:
        name = name[:60]
    if name not in bpy.data.materials.keys() or overwrite:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True

        # TODO: If language is set to slovensky, this will fail
        # seems to not be issue for other languages tested so far
        # bsdf = mat.node_tree.nodes["Principled BSDF"]
        for node in mat.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                bsdf = node
                break

        # Set base color
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)

        # # Connect alpha
        # a = mat.node_tree.nodes["Principled BSDF"].inputs["Alpha"]
        # mat.node_tree.links.new(sn.outputs["Alpha"], a)

        vcol = mat.node_tree.nodes.new(type="ShaderNodeVertexColor")
        vcol.location = [-400.0, 300.0]
        vcol.layer_name = "Colors"

        if link_vertex_color:
            mat.node_tree.links.new(vcol.outputs[0], bsdf.inputs[0])
    else:
        mat = bpy.data.materials[name]

    # mat.blend_method = "BLEND"
    # mat.shadow_method = "CLIP"
    # mat.node_tree.nodes["Image Texture"].image = image
    return mat


# ---------------------------------------------------------------------------
# Object / transform helpers
# ---------------------------------------------------------------------------


def scalemat(mat, sl):
    scaling = np.zeros_like(mat)
    scaling[np.diag_indices(4)] = sl
    # print(scaling)
    return np.matmul(scaling, mat)


def obj_unlink_all(obj):
    """Unlink object from all collections"""
    old_col = obj.users_collection

    # bugfix: not in master collection bug
    # collection_name.objects.unlink(obj)
    if len(old_col) > 0:
        for c in old_col:
            c.objects.unlink(obj)


def calculate_detail_level(dlev):
    """Angular deflection, Linear deflection"""
    if dlev < 100:
        l_def = 100.0 / float(dlev)
    else:
        l_def = (100.0 / float(dlev)) ** 2.0
    return 0.8, l_def


def set_obj_matrix_world(obj, mtx):
    """
    Copy Numpy matrix into Blender matrix
    """
    for row in range(mtx.shape[0]):
        for col in range(mtx.shape[1]):
            obj.matrix_world[row][col] = mtx[row][col]


def create_new_obj_with_mesh(name, set_active=True):
    """
    Create new empty object and mesh, link them, and optionally set to active
    """
    empty_mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, empty_mesh)
    bpy.context.collection.objects.link(obj)
    if set_active:
        bpy.context.view_layer.objects.active = obj
    return obj


def choose_hierarchy_types(htypes):
    """
    Return hierarchy types selection from input string
    """
    hierarchy_flat = False
    hierarchy_tree = False
    hierarchy_empties = False

    if htypes == "FLAT_AND_TREE":
        hierarchy_flat = True
        hierarchy_tree = True
    elif htypes == "TREE":
        hierarchy_tree = True
    elif htypes == "FLAT":
        hierarchy_flat = True
    elif htypes == "EMPTIES":
        hierarchy_empties = True
    else:
        assert False, "Invalid input parameter"

    return hierarchy_flat, hierarchy_tree, hierarchy_empties


def transform_to_up(up, chosen_objects, scale, to_cursor=True):
    """
    Set all chosen_objects transforms <up>["X", "Y", "Z"] as up
    Optionally move to cursor <to_cursor>
    Set scale to scale
    """

    # transforms and processing of objects
    # bpy.ops.object.select_all(action="DESELECT")

    cursor_pos = bpy.context.scene.cursor.location

    # up
    # up_as = self.up_as
    up_axis = {"X": 0, "Y": 1, "Z": 2}[up]

    # forward
    # fw_as = self.prg.fw_as
    # fw_axis = {"X": 0, "Y": 1, "Z": 2}[fw_as[0]]

    for obj in chosen_objects:
        # up, forward
        mat = np.array(obj.matrix_world)

        # blender default: Y(1) = forward, Z(2) = up
        if up_axis != 2:
            # if negate axis, do mirror
            # if up_as[1] == "N":
            #     dg = [1, 1, 1, 1]
            #     dg[up_axis] = -1
            #     mat = _scalemat(mat, dg)

            mat[[up_axis, 2]] = mat[[2, up_axis]]
            mat[up_axis] *= -1

        # scale
        mat = scalemat(mat, [*([scale] * 3), 1])

        # move to cursor position
        mat[0][3] += cursor_pos.x
        mat[1][3] += cursor_pos.y
        mat[2][3] += cursor_pos.z

        # apply
        set_obj_matrix_world(obj, mat)

    # Apply scale
    # for obj in created_objs:
    #     # Apply object scale
    #     obj.select_set(True)
    #     bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    #     obj.select_set(False)

    # for obj in created_objs:
    #     obj.select_set(True)


def freeze_matrix(objs):
    """Bake object scale into mesh data so transforms stay clean.
    Handles linked (instanced) mesh data by transforming the shared data once."""
    identity_vec = Vector((1, 1, 1))
    for o in objs:
        mat = Matrix()
        mat[0][0], mat[1][1], mat[2][2] = o.matrix_world.to_scale()
        # Only datablocks exposing transform() can have their scale baked in
        # (meshes). For others (e.g. curves) the scale stays on the object.
        if o.data and hasattr(o.data, "transform"):
            if o.data.users == 1:
                o.data.transform(mat)
                o.matrix_world = o.matrix_world.normalized()
            elif o.scale != identity_vec:
                instance_objs = [x for x in objs if x.data == o.data]
                first_obj = instance_objs[0]
                first_obj.data.transform(mat)
                for rest_obj in instance_objs:
                    rest_obj.matrix_world = rest_obj.matrix_world.normalized()


def shape_size(shp):
    bb = AIS_Shape(shp).BoundingBox()
    if bb.IsVoid():
        return 1.0
    diag = (bb.CornerMax().Distance(bb.CornerMin())) / 100000
    return diag


# ---------------------------------------------------------------------------
# Edge / mesh topology helpers (numpy sharp / seam detection)
# ---------------------------------------------------------------------------


def vert_of_edges(mesh: bpy.types.Mesh):
    edge_verts = np.empty(len(mesh.edges) * 2, dtype=np.int32)
    mesh.edges.foreach_get("vertices", edge_verts)
    edge_verts = edge_verts.reshape(-1, 2)  # transpose

    return edge_verts


def vert_coordinates_of_edges(mesh: bpy.types.Mesh, edge_verts):
    # Vert positions
    coords = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", coords)
    coords = coords.reshape(-1, 3)

    # Edge verts positions
    vert_co_0 = np.empty((len(mesh.edges), 3), dtype=np.float32)
    vert_co_1 = np.empty((len(mesh.edges), 3), dtype=np.float32)
    for i in range(len(mesh.edges)):
        vert_co_0[i] = coords[edge_verts[i, 0]]
        vert_co_1[i] = coords[edge_verts[i, 1]]

    return vert_co_0, vert_co_1


def faces_of_edges(mesh: bpy.types.Mesh):
    # Loop data
    loop_edge = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("edge_index", loop_edge)

    # Poly data
    loop_total = np.empty(len(mesh.polygons), dtype=np.int32)
    loop_start = np.empty(len(mesh.polygons), dtype=np.int32)
    mesh.polygons.foreach_get("loop_total", loop_total)
    mesh.polygons.foreach_get("loop_start", loop_start)

    # loop index -> poly index
    loop_to_poly = np.repeat(np.arange(len(mesh.polygons)), loop_total)

    # edge index -> list of poly indices
    faces_of_edges = [[] for _ in range(len(mesh.edges))]
    for loop_i, edge_i in enumerate(loop_edge):
        faces_of_edges[edge_i].append(loop_to_poly[loop_i])

    return faces_of_edges


# ---------------------------------------------------------------------------
# Addon preferences helpers
# ---------------------------------------------------------------------------


class FakeAddonPreferences:
    """Fallback preferences used when running in background / headless mode."""
    build_materials = True
    hack_skip_zero_solids = False
    simpler_parameters = True
    import_curves = False


def GetAddonPreferences(context):
    """Returns addon preferences; falls back to defaults in headless/background mode."""
    if bpy.app.background:
        return FakeAddonPreferences()
    return context.preferences.addons[__package__].preferences    #    @UndefinedVariable


# ---------------------------------------------------------------------------
# Vertex-color / material mesh update (TriMesh path)
# ---------------------------------------------------------------------------


def bpy_update_object_data(
    objdata, vcol_name, colors, uvs, norms, mat_names, build_materials=True
):
    # Geometry is already written into objdata (TriMesh.add_to_mesh) and edges
    # are marked via mark_edges(); here we only apply custom split normals, then
    # vertex colors and material indices via the faster mesh-level foreach_set
    # API, which avoids N_tris*3 Python-level BMesh attribute writes.
    if len(norms) > 0:
        objdata.normals_split_custom_set(np.array(norms))

    if not colors:
        return

    n_loops = len(colors)  # = N_tris * 3
    n_faces = n_loops // 3

    # All 3 loops of a triangle share the same per-face color/mat_name.
    # Build per-face RGBA once, then tile to loops with a single numpy repeat.
    face_rgba = np.empty((n_faces, 4), dtype=np.float32)
    face_rgba[:, 3] = 1.0

    mat_index_arr = np.zeros(n_faces, dtype=np.int32) if build_materials else None

    if build_materials:
        obj_mats = {ob_mat.name: obi for obi, ob_mat in enumerate(objdata.materials)}
        mat_counter = 0

    for fi in range(n_faces):
        col = colors[fi * 3]
        mn = mat_names[fi * 3]
        if col is not None and col[0] >= 0.0:
            r, g, b = float(col[0]), float(col[1]), float(col[2])
        else:
            r, g, b = 0.5, 0.5, 0.5
            mn = None
        face_rgba[fi, 0] = r
        face_rgba[fi, 1] = g
        face_rgba[fi, 2] = b

        if build_materials:
            if mn is None:
                mn = "STEP_" + "{:02x}{:02x}{:02x}".format(
                    int(r * 255), int(g * 255), int(b * 255)
                )
            if mn not in bpy.data.materials:
                add_material(mn, (r, g, b), link_vertex_color=False)
            if mn not in obj_mats:
                obj_mats[mn] = mat_counter
                objdata.materials.append(bpy.data.materials[mn])
                mat_counter += 1
            mat_index_arr[fi] = obj_mats[mn]

    # Tile face RGBA to per-loop data and write at C level
    flat_rgba = np.repeat(face_rgba, 3, axis=0).ravel()
    color_layer = objdata.vertex_colors.get(vcol_name)
    if color_layer is None:
        color_layer = objdata.vertex_colors.new(name=vcol_name)
    color_layer.data.foreach_set("color", flat_rgba)

    if build_materials:
        objdata.polygons.foreach_set("material_index", mat_index_arr)
