import numpy as np
import json
import argparse

parser = argparse.ArgumentParser(description='Argument parser')
parser.add_argument('--src_json_file', dest='src_json_file', type=str, default='/mnt/d/research/nerfstudio/nerfstudio/data/clinical_3/cam_path/transforms_train.json',
                    help='path to the source json file')
parser.add_argument('--dest_json_file', dest='dest_json_file', type=str, default='/mnt/d/research/nerfstudio/nerfstudio/data/clinical_3/cam_path/transforms_train_right.json',
                    help='path to the destination json file')
parser.add_argument('--shift', type=float, help='shift in y direction', default= 0.1)
args = parser.parse_args()

def get_translated_matrix(transform_matrix: list, shift: float):
    transform_matrix = np.array(transform_matrix + [[0, 0, 0, 1]])  # 添加第四行 [0, 0, 0, 1]
    shift_matrix = np.array([[1.0, 0.0, 0.0, 0],
                             [0.0, 1.0, 0.0, shift],
                             [0.0, 0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0]])
    result = np.linalg.inv(shift_matrix @ np.linalg.inv(transform_matrix))
    return result[:3, :4].tolist()  # 返回前三行

# 打开源JSON文件
with open(args.src_json_file, 'r') as src_file:
    data = json.load(src_file)

# 遍历JSON数据中的每一帧
for frame in data:
    # 计算新的transform矩阵
    t_matrix = frame["transform"]
    frame["transform"] = get_translated_matrix(transform_matrix=t_matrix, shift=args.shift)

# 将修改后的JSON数据写入目标文件
with open(args.dest_json_file, 'w') as dest_file:
    json.dump(data, dest_file, indent=2)

print(f"处理后的JSON已保存至 {args.dest_json_file}")