import subprocess
import pandas as pd

# sing = [100]
# feature_list = [80, 120, 160, 200]
# for feature_cnt in feature_list:
#     for sing_cnt in sing:
#         for num in range(1, 51):
#             print(feature_cnt, sing_cnt, num)
#             subprocess.call(["python", "main.py", str(feature_cnt), str(sing_cnt), str(num), "Qian_Li_data"])



sing = [0, 50, 100]
feature_list = [40, 80, 120, 160, 200]
for feature_cnt in feature_list:
    for sing_cnt in sing:
        for num in range(11, 16):
            print(feature_cnt, sing_cnt, num)
            subprocess.call(["python", "main.py", str(feature_cnt), str(sing_cnt), str(num), "data"])



# df = pd.read_csv("./result")

# result = pd.DataFrame(columns=["features", "sing", "num", "acc", "auc", ""])

# for sing_cnt in sing:
#     for feature_cnt in feature_list:
#         df[df["features"] == feature_cnt and df["sing"] == sing_cnt]