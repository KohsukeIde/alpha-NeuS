import open3d
import numpy as np

def distance_count(gt, pred, number_of_points):
    if np.asarray(gt.triangles).shape[0] == 0:
        pc1 = open3d.geometry.PointCloud(gt.vertices)
    else:
        pc1 = gt.sample_points_uniformly(number_of_points=number_of_points)
    if np.asarray(pred.triangles).shape[0] == 0:
        pc2 = open3d.geometry.PointCloud(pred.vertices)
    else:
        pc2 = pred.sample_points_uniformly(number_of_points=number_of_points)
    re1 = np.asarray(pc1.compute_point_cloud_distance(pc2)).mean()
    re2 = np.asarray(pc2.compute_point_cloud_distance(pc1)).mean()
    return [re1,re2,(re1+re2)/2]

dataset = "jar"
dir1 = "final_data/" + dataset + "/base_iso_0.0/" + "meshes/" + "00300000.ply"              # neus_iso_0.0
#dir2 = "final_data/" + dataset + "/base_iso_0.0/" + "meshes/" + "00300000_0.005.ply"       # neus_iso_0.005
dir2 = "final_data/" + dataset + "/base_iso_0.0/" + "udf_meshes/" + "00300000.ply"          # dcudf
dir3 = "alphasurf/" + f'{dataset}_mesh_0.003_trimmed.ply'                                   # alphasurf
dir4 = "NeUDF/0.018/" + f'{dataset}.ply'                                                    # NeUDF

gt = open3d.io.read_triangle_mesh("final_data/gtmesh/" + dataset + ".ply")
pred_1 = open3d.io.read_triangle_mesh(dir1)
pred_2 = open3d.io.read_triangle_mesh(dir2)
pred_3 = open3d.io.read_triangle_mesh(dir3)
pred_4 = open3d.io.read_triangle_mesh(dir4)

# 计算模型的边界框
vertices = np.asarray(gt.vertices)
min_bound = vertices.min(axis=0)
max_bound = vertices.max(axis=0)
model_size = max_bound - min_bound

# 确定模型的最长边和相应的缩放比例
max_size = max(model_size)
scale_factor = 1.0 / max_size
print("scale_factor: ", scale_factor)

# 缩放顶点数据
scaled_vertices = np.asarray(gt.vertices - min_bound) * scale_factor
gt.vertices = open3d.utility.Vector3dVector(scaled_vertices)
scaled_vertices = np.asarray(pred_1.vertices - min_bound) * scale_factor
pred_1.vertices = open3d.utility.Vector3dVector(scaled_vertices)
scaled_vertices = np.asarray(pred_2.vertices - min_bound) * scale_factor
pred_2.vertices = open3d.utility.Vector3dVector(scaled_vertices)
scaled_vertices = np.asarray(pred_3.vertices - min_bound) * scale_factor
pred_3.vertices = open3d.utility.Vector3dVector(scaled_vertices)
scaled_vertices = np.asarray(pred_4.vertices - min_bound) * scale_factor
pred_4.vertices = open3d.utility.Vector3dVector(scaled_vertices)


number_of_points = 100000

print(distance_count(gt, pred_1, number_of_points))
print(distance_count(gt, pred_2, number_of_points))
print(distance_count(gt, pred_3, number_of_points))
print(distance_count(gt, pred_4, number_of_points))
