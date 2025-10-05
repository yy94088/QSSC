import numpy as np
import json

edge_embeds = np.load("data/prone/hprd_edge.emb.npy")

# 构造查找 key
labels_src = [25, 1]
labels_dst = [25, 1]
key = str((6, 48))  # JSON 中 key 会变成字符串

# 加载查找表
with open("data/prone/hprd_edge_index_map.json", "r") as f:
    edge_index_map = json.load(f)

if key in edge_index_map:
    for idx in edge_index_map[key]:
        print(f"边嵌入[{idx}]:", edge_embeds[idx])
else:
    print("找不到匹配的边")
