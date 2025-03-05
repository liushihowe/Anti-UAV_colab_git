
label_dir = 'D:\study\2025_gradproj\数据集\分好的\3_test'; % 路径

% 获取所有符合命名规则的.mat文件
file_list = dir(fullfile(label_dir, 'IR_*_LABELS.mat')); 

% 初始化空数组存储gTruth对象
gTruth = [];

% 批量加载所有标签文件
for i = 1:length(file_list)
    file_path = fullfile(file_list(i).folder, file_list(i).name);
    loaded_data = load(file_path);
    gTruth = [gTruth; loaded_data.gTruth];
end

mkdir('Training_data_IR')
addpath('Training_data_IR');

%% Produce the training dataset
onesMatrix = ones(1, length(gTruth));  % 创建一个 length(gTruth) x 2 的全一数组

trainingData = objectDetectorTrainingData(gTruth,...
    'SamplingFactor', 10 * onesMatrix, ...
    'WriteLocation','Training_data_IR');

%% 转换为 YOLO 格式并生成 txt 文件
% 定义类别映射 (名称 → ID)
class_names = {'AIRPLANE', 'BIRD', 'DRONE', 'HELICOPTER'};
class_dict = containers.Map(class_names, 0:3); % 名称到ID的映射

% 创建 YOLO 格式目录结构
yolo_dir = 'YOLO_Dataset';
mkdir(fullfile(yolo_dir, 'images', 'train')); % 训练集图片
mkdir(fullfile(yolo_dir, 'labels', 'train')); % 训练集标签

% 遍历每张图片
for i = 1:height(trainingData)
    % 读取图片信息
    img_path = trainingData.imageFilename{i};
    [~, img_name, ~] = fileparts(img_path);
    img = imread(img_path);
    [img_height, img_width, ~] = size(img);
    
    % 创建 YOLO 标签文件路径
    txt_path = fullfile(yolo_dir, 'labels', 'train', [img_name '.txt']);
    
    % 打开文件准备写入
    fid = fopen(txt_path, 'w');
    
    % 遍历所有类别
    for cls_idx = 1:length(class_names)
        bboxes = trainingData.(class_names{cls_idx}){i};
        if ~isempty(bboxes)
            % 处理每个边界框 (MATLAB 可能返回多行)
            for j = 1:size(bboxes, 1)
                % 获取绝对坐标
                x = bboxes(j,1);
                y = bboxes(j,2);
                w = bboxes(j,3);
                h = bboxes(j,4);
                
                % 转换为归一化坐标
                x_center = (x + w/2) / img_width;
                y_center = (y + h/2) / img_height;
                norm_w = w / img_width;
                norm_h = h / img_height;
                
                % 写入文件 (格式: class_id x_center y_center width height)
                fprintf(fid, '%d %.6f %.6f %.6f %.6f\n',...
                    class_dict(class_names{cls_idx}),...
                    x_center, y_center, norm_w, norm_h);

                % fprintf(fid, '%d %.6f %.6f %.6f %.6f\n',...
                %     0,...
                %     x_center, y_center, norm_w, norm_h);

            end
        end
    end
    
    fclose(fid);
    
    % 复制图片到 YOLO 目录
    copyfile(img_path, fullfile(yolo_dir, 'images', 'train', [img_name '.jpg']));
end


%% 生成 data.yaml 配置文件
yaml_content = {
    'train: ../YOLO_Dataset/images/train';
    'val: ../YOLO_Dataset/images/val    # 根据需要添加验证集';
    'test:  # 测试集路径';
    '';
    'nc: 4';
    'names: [''AIRPLANE'', ''BIRD'', ''DRONE'', ''HELICOPTER'']'
};

fid = fopen(fullfile(yolo_dir, 'data.yaml'), 'w');
fprintf(fid, '%s\n', yaml_content{:});
fclose(fid);

disp('YOLO 格式数据集已生成！');