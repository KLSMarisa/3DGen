import diffusers
import os

# diffusers.__file__ 指向库的 __init__.py 文件
# 使用 os.path.dirname() 来获取该文件所在的目录路径
library_path = os.path.dirname(diffusers.__file__)

print(f"Diffusers library path is: {library_path}")