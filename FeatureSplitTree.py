# import random
# import math
#
# class Node:
#     def __init__(self, point, left=None, right=None, axis=0):
#         self.point = point
#         self.left = left
#         self.right = right
#         self.axis = axis
#
# def build_tree(points, depth=0):
#     if not points:
#         return None
#
#     k = len(points[0])  # 数据点的维度
#     axis = depth % k  # 选择维度
#
#     points.sort(key=lambda x: x[axis])
#     median = len(points) // 2
#
#     return Node(
#         point=points[median],
#         left=build_tree(points[:median], depth + 1),
#         right=build_tree(points[median + 1:], depth + 1),
#         axis=axis
#     )
#
# def print_tree(node, depth=0, prefix="Root: "):
#     if not node:
#         return
#     point_str = " ".join(f"{x:016x}" for x in node.point)
#     print(" " * (depth * 4) + prefix + f"Point [{point_str}] (Axis {node.axis})")
#     if node.left:
#         print_tree(node.left, depth + 1, prefix="L--- ")
#     if node.right:
#         print_tree(node.right, depth + 1, prefix="R--- ")
#
# def hamming_distance(point1, point2):
#     total_distance = 0
#     for i in range(len(point1)):
#         total_distance += bin(point1[i] ^ point2[i]).count('1')
#     return total_distance
#
#
# def nearest_neighbor(node, point, depth=0, best=None, best_dist=float('inf')):
#     if node is None:
#         return best, best_dist
#
#     k = len(point)
#     axis = depth % k
#
#     next_best = None
#     next_branch = None
#
#     if best is None or hamming_distance(point, node.point) < best_dist:
#         next_best = node.point
#         best_dist = hamming_distance(point, node.point)
#     else:
#         next_best = best
#
#     point_str = " ".join(f"{x:016x}" for x in point)
#     node_point_str = " ".join(f"{x:016x}" for x in node.point)
#     next_best_str = " ".join(f"{x:016x}" for x in next_best)
#
#     print(f"Depth {depth}: Comparing point [{point_str}] with node [{node_point_str}] (Distance: {hamming_distance(point, node.point)})")
#     print(f"Current best: [{next_best_str}] (Distance: {best_dist})")
#
#     if point[axis] < node.point[axis]:
#         next_branch = node.left
#         opposite_branch = node.right
#     else:
#         next_branch = node.right
#         opposite_branch = node.left
#
#     next_best, best_dist = nearest_neighbor(next_branch, point, depth + 1, next_best, best_dist)
#
#     if (point[axis] - node.point[axis]) ** 2 < best_dist:
#         next_best, best_dist = nearest_neighbor(opposite_branch, point, depth + 1, next_best, best_dist)
#
#     return next_best, best_dist
#
# 生成100个随机数据点
import time


def generate_random_points(n, dimensions):
    return [[random.randint(0x0000000000000000, 0xffffffffffffffff) for _ in range(dimensions)] for _ in range(n)]
#
# # 示例使用
# points = generate_random_points(100000, 3)
# tree = build_tree(points)
#
#
# # 随机生成查询点
# query_point = [random.randint(0x7676776677666666, 0x7777777777777777) for _ in range(3)]
# nearest, dist = nearest_neighbor(tree, query_point)
# print(f"\nQuery point: [{ ' '.join(f'{x:016x}' for x in query_point)}]")
# print(f"Nearest neighbor: [{ ' '.join(f'{x:016x}' for x in nearest)}] (Distance: {dist})")
import random
import math
from collections import Counter

class Node:
    def __init__(self, point, left=None, right=None, axis=0, index=None):
        self.point = point
        self.left = left
        self.right = right
        self.axis = axis
        self.index = index  # 添加索引属性，用于存储数据点的编号

def build_tree(points, depth=0, index=0):  # 修改build_tree函数以接收索引参数
    if not points:
        return None

    k = len(points[0])  # 数据点的维度
    axis = depth % k  # 选择维度

    points.sort(key=lambda x: x[axis])
    median = len(points) // 2

    return Node(
        point=points[median],
        left=build_tree(points[:median], depth + 1, index),
        right=build_tree(points[median + 1:], depth + 1, index + median + 1),
        axis=axis,
        index=index  # 存储当前节点的索引
    )

def hamming_distance(point1, point2):
    total_distance = 0
    for i in range(len(point1)):
        total_distance += bin(point1[i] ^ point2[i]).count('1')
    return total_distance

def nearest_neighbor(node, point, depth=0, best_dist=float('inf'), best_index=None):
    if node is None:
        return best_index, best_dist

    k = len(point)
    axis = depth % k

    if best_index is None or hamming_distance(point, node.point) < best_dist:
        best_index = node.index
        best_dist = hamming_distance(point, node.point)

    if point[axis] < node.point[axis]:
        next_branch = node.left
        opposite_branch = node.right
    else:
        next_branch = node.right
        opposite_branch = node.left

    best_index, best_dist = nearest_neighbor(next_branch, point, depth + 1, best_dist, best_index)

    if (point[axis] - node.point[axis]) ** 2 < best_dist:
        best_index, best_dist = nearest_neighbor(opposite_branch, point, depth + 1, best_dist, best_index)

    return best_index, best_dist

# def find_most_common_neighbors(tree, points, k):
#     neighbors = []
#     for i, point in enumerate(points):
#         nearest_index, _ = nearest_neighbor(tree, point)
#         neighbors.append(nearest_index)
#     # if len(neighbors)<=5:
#     #     print(neighbors)
#     #     return neighbors
#     counter = Counter(neighbors)
#     most_common = counter.most_common(k)
#     print('most_common:',most_common)
#     return most_common

def print_tree(node, depth=0, prefix="Root: "):
    if not node:
        return
    point_str = " ".join(f"{x:016x}" for x in node.point)
    print(" " * (depth * 4) + prefix + f"Point [{point_str}] (Axis {node.axis})")
    if node.left:
        print_tree(node.left, depth + 1, prefix="L--- ")
    if node.right:
        print_tree(node.right, depth + 1, prefix="R--- ")

def build_triple_tree(points):
    tree_first = build_tree(points)
    tree_second = build_tree(points, depth=1)
    tree_third = build_tree(points, depth=2)
    return tree_first,tree_second,tree_third

def find_most_common_neighbors_triple(tree1,tree2,tree3,points,k):
    neighbors = []
    for i, point in enumerate(points):
        nearest_index1, _ = nearest_neighbor(tree1, point)
        nearest_index2, _ = nearest_neighbor(tree2, point)
        nearest_index3, _ = nearest_neighbor(tree3, point)
        if nearest_index1 not in neighbors:
            neighbors.append(nearest_index1)
        if nearest_index2 not in neighbors:
            neighbors.append(nearest_index2)
        if nearest_index3 not in neighbors:
            neighbors.append(nearest_index3)
    counter = Counter(neighbors)
    most_common = counter.most_common(k)
    return most_common
# # 生成100个随机数据点
# random.seed(time.time())
# points = generate_random_points(100, 3)
# tree = build_tree(points)
#
# # 对每个数据点执行最近邻查询并统计出现次数最多的k个点的编号
# most_common_neighbors = find_most_common_neighbors(tree, points, 5)
# for i in range(len(most_common_neighbors)):
#     print(most_common_neighbors[i][0])
