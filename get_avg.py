import pandas as pd
import sys

file_name = sys.argv[1]

df = pd.read_csv(file_name)

gb = df.groupby(["features", "sing"])

df_result = pd.DataFrame(columns=["features", "sing", "acc", "auc"])

for gb_i in gb:
    print(gb_i[0])
    df_i = gb_i[1]
    df_result = df_result.append({
        "features" : gb_i[0][0], 
        "sing" : gb_i[0][1], 
        "acc" : df_i.mean()["acc"], 
        "auc" : df_i.mean()["auc"]
    }, ignore_index=True)

df_result.to_csv("mean_" + file_name, index=False)