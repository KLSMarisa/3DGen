import pandas as pd
import random
import json



def create_final_dataset(csv_path, output_json_path):
    # Read the original dataset
    data = pd.read_csv(csv_path)
    print(f"原始行数: {len(data)}")
    # Ensure the "aesthetic" column exists
    if "aesthetic" not in data.columns:
        raise ValueError("The CSV file must contain an 'aesthetic' column.")
    
    # Find a classification column to use for filtering (exclude 'scene')
    possible_cls_cols = ["category"]
    class_col = next((c for c in possible_cls_cols if c in data.columns), None)
    if class_col is None:
        raise ValueError(f"The CSV file must contain a classification column (one of {possible_cls_cols}).")
    
    # Filter out rows where classification == 'scene' (case-insensitive) or category is empty/null
    before_filter_count = len(data)
    col_series = data[class_col].fillna("").astype(str)
    non_empty_mask = col_series.str.strip() != ""
    not_scene_mask = ~col_series.str.lower().eq("scene")
    data = data[not_scene_mask].copy()
    
    after_filter_count = len(data)
    print(f"筛选后（排除 空 category 与 '{class_col}' == 'scene'）行数: {after_filter_count} (原始 {before_filter_count})")
    
    # Ensure the path-related columns exist
    required_columns = ["white_part", "dataset_id"]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"The CSV file must contain a '{col}' column.")

    # Take the top 200k high-scoring rows from the filtered dataset
    top_200k = data.sort_values(by="aesthetic", ascending=False).head(50000)
    if len(top_200k) == 0:
        raise ValueError("No rows remain after filtering; cannot build dataset.")

    # 输出 top200k 最后一项的美学得分（即第200000名的得分，如果不足200000则输出最后一项得分）
    last_score = top_200k["aesthetic"].iloc[-1]
    print(f"top200k 最后一项的美学得分: {last_score}")
    # Randomly sample 50k rows from the filtered dataset
    random_100k = data.sort_values(by="aesthetic", ascending=False)[200000:].sample(n=min(100000, len(data)), random_state=42)

    # Combine the top 200k and random 100k rows
    combined_data = top_200k#pd.concat([top_200k, random_100k]).drop_duplicates()
    #combined_data= top_50k
    # Create the final dataset with specific data paths
    final_dataset = combined_data.apply(
        lambda row: f"/mnt/hdd1/caixiao/data/objaverse_1.0/hf-objaverse-v1/glbs/{row[required_columns[0]]}/{row[required_columns[1]]}.glb", axis=1
    ).tolist()
    with open(output_json_path, "w") as json_file:
        json.dump(final_dataset, json_file)
    #path_list = combined_data.apply(
    #    lambda row: f"/mnt/hdd1/caixiao/data/pv_views/{row[required_columns[0]]}/{row[required_columns[1]]}", axis=1
    #).tolist()
    
    # Save the final dataset as a JSON list
    print(len(final_dataset))
    #with open('combined_path.json', "w") as json_file:
    #    json.dump(path_list, json_file)
    
    print(f"Final dataset saved to {output_json_path}")

# Example usage:
create_final_dataset("/mnt/hdd1/caixiao/data/gt23d-bench/all_flag_8.csv", "high_score_50k.json")
