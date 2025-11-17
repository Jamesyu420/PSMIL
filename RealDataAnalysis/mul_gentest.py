import os
import subprocess
file_list = os.listdir("/database/datasets/CAMELYON16/testing/images")
remove_list = ['test_049.tif','test_114.tif']
file_list = [item for item in file_list if item not in remove_list]


ff = os.listdir("/teams/WSIresult_1727165526/RealDataResult/512_75/testXAY/")
ff = [item[:-3]+'tif' for item in ff]
file_l = [item for item in file_list if item not in ff]
print(len(file_l))

json_list = os.listdir('/mnt/GMM_for_MIL_codes/DataAnalysis_3/data/test/json_annotations/')

N = len(file_list)

i = -1
for file in file_l:
    i += 1
    subprocess.run(['python', './sin_gentest.py', "-filename", file, "-num", str(i)])
