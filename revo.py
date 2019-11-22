import subprocess

sing = [0, 50, 100]
feature_list = [40, 80, 120, 160, 200]
nums = range(11, 21)
for feature_cnt in feature_list:
    for sing_cnt in sing:
        for num in range(11, 21):
            print(feature_cnt, sing_cnt, num)
            subprocess.call(["python", "main.py",str(sing_cnt),str(feature_cnt),str(num)])
