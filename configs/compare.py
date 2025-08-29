import json

def compare_json_files(file_a, file_b, output_file):
    """
    比较两个 JSON 文件，找出只存在于第一个文件中，而不存在于第二个文件中的项目。

    参数:
    file_a (str): 第一个 JSON 文件的路径。
    file_b (str): 第二个 JSON 文件的路径。
    output_file (str): 用于保存差异结果的 JSON 文件的路径。
    """
    try:
        # 读取第一个 JSON 文件
        with open(file_a, 'r', encoding='utf-8') as f:
            data_a = json.load(f)

        # 读取第二个 JSON 文件
        with open(file_b, 'r', encoding='utf-8') as f:
            data_b = json.load(f)

        # 确保读取的数据是列表格式
        if not isinstance(data_a, list) or not isinstance(data_b, list):
            print("错误：JSON 文件的内容必须是列表。")
            return
        print('len(data_a):',len(data_a))
        print('len(data_b):',len(data_b))
        # 将列表转换为集合以提高比较效率
        set_a = set(data_a)
        set_b = set(data_b)
        # 计算差集 (在 a 中但不在 b 中的元素)
        difference = set_a.difference(set_b)

        # 将结果转换回列表
        result_list = list(difference)

        # 将结果写入输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_list, f, indent=4, ensure_ascii=False)

        print(f"比较完成！结果已保存到 {output_file}")
        print(f"在 '{file_a}' 中但不在 '{file_b}' 中的项目有 {len(result_list)} 个。")

    except FileNotFoundError as e:
        print(f"错误：找不到文件 {e.filename}")
    except json.JSONDecodeError:
        print("错误：JSON 文件格式不正确。")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == '__main__':
    # 定义输入和输出文件名
    file_a = 'rgb2.json'
    file_b = '/mnt/hdd1/data/select_data1.json'
    output_file = 'diff.json'

    # 调用函数进行比较
    compare_json_files(file_a, file_b, output_file)