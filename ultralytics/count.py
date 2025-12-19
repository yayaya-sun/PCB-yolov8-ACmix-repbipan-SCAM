import os
def labels_count(root):
    '''
    统计数据集中每个类别的数量
    args:
        root:yolog格式数据集的label路径
    return:
        labels_count:数据集中每个类别的数量
    '''
    labels_count = {}
    for dir in os.listdir(root):
        labels_dir_count = {}
        if dir in ['train', 'valid', 'test']:
            label_dir = os.path.join(root, dir)
            label_files = os.listdir(label_dir)  # 展示目标文件夹下所有的文件名
            label_files = list(filter(lambda x: x.endswith('.txt'), label_files))  # 取到所有以.txt结尾的yolo格式文件
            for label_file in label_files:
                with open(os.path.join(label_dir, label_file), 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    line = line.strip().split()
                    class_name = line[0]
                    if class_name in labels_count:
                        labels_count[class_name] += 1
                    else:
                        labels_count[class_name] = 1
                    if class_name in labels_dir_count:
                        labels_dir_count[class_name] += 1
                    else:
                        labels_dir_count[class_name] = 1
            print(f'{dir} done!', 'labels_count:', labels_dir_count)

    return labels_count

if __name__ == '__main__':
    root = r'/root/autodl-tmp/labels' # dataset为你的YOLOv8格式的数据集hhhh
    labels_cnt = labels_count(root)
    print('all sets done!   labels_count:', labels_cnt)
