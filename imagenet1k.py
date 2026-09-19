

#############################################
##   这个代码用于准备Imagenet1K的开放集测试。
#    step 1 请将ImageNet1k.tar 下载到/root/data/ImageNet1k
#    step 2  解压缩ImageNet1k.tar到/root/data/ImageNet1k_open/
#    step 3  运行38-81行，请注释掉81行以后的代码
#    step 4  再次解压缩ImageNet1k.tar到/root/data/ImageNet1k_close/
#    step 4  运行81行以后代码，请注释掉38-81行代码
#############################################

import os
import shutil

def list_directories(path):
    # 获取指定目录下的所有文件和文件夹
    entries = os.listdir(path)
    # 初始化一个列表来存储文件夹名称
    directories = []
    # 遍历每个条目
    for entry in entries:
        # 构建完整的路径
        full_path = os.path.join(path, entry)
        # 检查是否为目录
        if os.path.isdir(full_path):
            # 如果是目录，添加到列表中
            directories.append(entry)
    return directories




directory_path = '/root/data/ImageNet1k_open/train'  # 替换为你的目标目录路径
directories_list = list_directories(directory_path)
directories_list.sort()
print(directories_list)

print(len(list_directories(directory_path)))

##########################   第一次运行不注释，第二次运行注释掉
# #######################  test ：删除open中close
# directory_path = '/root/data/ImageNet1k_open/train'  # 替换为你的目标目录路径
# directories_list = list_directories(directory_path)
# directories_list.sort()
# print(directories_list)
#
# print(len(list_directories(directory_path)))
#
# directories_list = ["/root/data/ImageNet1k_open/train/" + x for x in directories_list]
#
# print(directories_list)
# print("====================================================================")
# delete_list1 = directories_list[500:]
# print(delete_list1)
#
#
# for path in delete_list1:
#     print("正在删除", path)
#     shutil.rmtree(path)
# #######################  test ：删除close中open
#
#
# #######################  test ：删除open中close
# directory_path = '/root/data/ImageNet1k_open/val'  # 替换为你的目标目录路径
# directories_list = list_directories(directory_path)
# directories_list.sort()
# print(directories_list)
#
# print(len(list_directories(directory_path)))
#
# directories_list = ["/root/data/ImageNet1k_open/val/" + x for x in directories_list]
#
# print(directories_list)
# print("====================================================================")
# delete_list1 = directories_list[500:]
# print(delete_list1)
#
#
# for path in delete_list1:
#     print("正在删除", path)
#     shutil.rmtree(path)
# #######################  test ：删除close中open
##########################   第一次运行不注释，第二次运行注释掉



##########################   第二次运行不注释，第一次运行注释掉
#######################  test ：删除close中open
directory_path = '/root/data/ImageNet1k_close/val'  # 替换为你的目标目录路径
directories_list = list_directories(directory_path)
directories_list.sort()
print(directories_list)

print(len(list_directories(directory_path)))

directories_list = ["/root/data/ImageNet1k_close/val/" + x for x in directories_list]

print(directories_list)
print("====================================================================")
delete_list1 = directories_list[:500]
print(delete_list1)


for path in delete_list1:
    print("正在删除", path)
    shutil.rmtree(path)
#######################  test ：删除close中open


#######################  train ：删除close中open
# 使用示例
directory_path = '/root/data/ImageNet1k_close/train'  # 替换为你的目标目录路径
directories_list = list_directories(directory_path)
directories_list.sort()
print(directories_list)

print(len(list_directories(directory_path)))

directories_list = ["/root/data/ImageNet1k_close/train/" + x for x in directories_list]

print(directories_list)
print("====================================================================")
delete_list1 = directories_list[:500]
print(delete_list1)


for path in delete_list1:
    print("正在删除", path)
    shutil.rmtree(path)
#######################  train ：删除close中open
##########################   第二次运行不注释，第一次运行注释掉