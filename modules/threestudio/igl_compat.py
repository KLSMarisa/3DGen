"""Compatibility shim: map old igl API to libigl 2.x names."""
import igl

fast_winding_number_for_meshes = igl.fast_winding_number
point_mesh_squared_distance = igl.point_mesh_squared_distance
read_obj = igl.readOBJ
