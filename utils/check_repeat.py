import json
from collections import Counter
import os
# 读取 JSON 文件
with open('/home/linzhuohang/3DGen/configs/multiview_300k.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
with open('/home/linzhuohang/3DGen/configs/rgb2.json', 'r', encoding='utf-8') as f:
    data2 = json.load(f)
data_filenames = {os.path.basename(item) for item in data}
data2_minus_data = [item for item in data2 if os.path.basename(item) not in data_filenames]
with open('/home/linzhuohang/3DGen/configs/rgb_multiview.json', 'w', encoding='utf-8') as out:
    json.dump(data2_minus_data, out, ensure_ascii=False, indent=2)
# 假设每个条目有 'filename' 字段
data.extend(data2)
filenames = [os.path.basename(item) for item in data]
#print(filenames)
# 统计文件名出现次数
counter = Counter(filenames)
print(len(counter.items()))
# 输出重复的文件名及其数量
repeat_files = {fname: count for fname, count in counter.items() if count > 1}
print(f"重复的文件名数量: {len(repeat_files)}")
cnt=len(repeat_files)
print(cnt)