import os
import json

def scan_dirs(path):
    dirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return json.dumps(dirs, indent=4)

if __name__ == '__main__':
    target_path = '/mnt/hdd3/linzhuohang/3DGen/gso_rendered'  # 替换成你要扫描的目录路径
    save_dir = 'configs/gso.json'
    json_output = scan_dirs(target_path)
    with open(save_dir, 'w') as f:
        f.write(json_output)
    #print(json_output)