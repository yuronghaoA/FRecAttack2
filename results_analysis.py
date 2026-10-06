import os
import re
import pandas as pd

win_size = 3
compile_str = re.compile( r'.*?, loss = (\d*?\.\d*?), HR@10 = (\d*?\.\d*?), nDCG@10 = (\d*\.\d*)')

def file_result_analysis(file_name):
    file_data = []
    with open (file_name,"r") as f:
        context = f.readlines()
        for line_data in context:
            data = re.findall(compile_str, line_data.rstrip('\n'))
            if data:
                data = list(map(float,data[0]))
                file_data.append(data)
    DF_file_data = pd.DataFrame(file_data,columns=["loss","HR@10","nDCG@10"])
    return DF_file_data[-3:][["HR@10","nDCG@10"]].mean()

def result_analysis(dir_path):
    file_list = os.listdir(dir_path)
    result_dict = {}
    result_dict_guard = {}

    for file_name in file_list:
        if "GuardCQ" in file_name:
            if "NoAttack" in file_name:
                NoAttack_guard_name = file_name
            result_dict_guard[file_name] = file_result_analysis(os.path.join(dir_path,file_name))
        else:
            if "NoAttack" in file_name:
                NoAttack_name = file_name
            result_dict[file_name] = file_result_analysis(os.path.join(dir_path,file_name))

    DF_result = pd.DataFrame(result_dict).T
    DF_result_guard = pd.DataFrame(result_dict_guard).T

    DF_result_ = (DF_result.loc[NoAttack_name] - DF_result) / DF_result.loc[NoAttack_name]
    DF_result_guard_ = (DF_result_guard.loc[NoAttack_guard_name]-DF_result_guard)/DF_result_guard.loc[NoAttack_guard_name]

    cprint("Parsing directory: " +dir_path)
    cprint2("Attack performance with no defences. (The higher the %, the better the attack performance.)")
    print(DF_result)
    print(DF_result_.drop(NoAttack_name).applymap(lambda x: "{:.2%}".format(x)))
    print("\n")
    cprint2("Attack performance with our GuardCQ defences. (The lower the %, the better the defense performance.)")
    print(DF_result_guard)
    print(DF_result_guard_.drop(NoAttack_guard_name).applymap(lambda x:"{:.2%}".format(x)))
    print("\n")

def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")
def cprint2(words: str):
    print(f"\033[0;31;40m{words}\033[0m")

def path_parse():
    dir1 = "./results"
    dir2 = os.listdir(dir1)
    dir2 = [os.path.join(dir1,dir) for dir in dir2]
    for dir in dir2:
        for dir_ in os.listdir(dir):
            dir_path = os.path.join(dir,dir_)
            result_analysis(dir_path)


if __name__ == "__main__":
    path_parse()