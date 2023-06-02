function lab5_3(filename)
    src = imread(filename);
    % 图像块的颜色范围阈值
    threshold = 0.55;
    % 四叉树分割
    qtree = qtdecomp(src, threshold, 2);
    blocks = zeros(size(qtree));
    
    % 区域分割，针对每个给定的块大小 dim，依次进行分割
    for dim = [256 128 64 32 16 8 4 2]
        % 计算图像中包含的该大小的块的数量
        numblocks = length(find(qtree == dim));
        % 如果图像中存在该大小的块，则进行处理
        if (numblocks > 0)        
            values = repmat(uint8(1), [dim dim numblocks]);
            values(2 : dim, 2 : dim, : ) = 0;
            % 将 values 矩阵中的块插入到 blocks 矩阵中
            blocks = qtsetblk(blocks, qtree, dim, values);
        end
    end
    
    % 得到分割结果
    output1 = src;
    output1(blocks == 1) = 255; % 图像边界
    
    % show
    subplot(1, 3, 1); imshow(src, []); title('原图像'); 
    subplot(1, 3, 2); imshow(output1, []); title('分割结果'); 
    
    % 为区域分割得到每个块进行标记
    i = 0;
    for dim = [256 128 64 32 16 8 4 2]
        [vals, r, c] = qtgetblk(src, qtree, dim);
        % 如果图像中存在该大小的块，则进行处理
        if ~isempty(vals)
            for j = 1 : length(r)
                i = i + 1;
                % 将该块内的所有像素的值设置为标记 i，以表示该像素点所在的分块
                blocks(r(j) : r(j) + dim - 1, c(j) : c(j) + dim - 1) = i;
            end
        end
    end
    
    % 将区域分割的块进行合并
    for j = 1 : i
        % 生成相邻分块的边界掩码，并检查其是否与当前分块相邻
        bound = boundarymask(blocks == j, 4) & (~(blocks == j));
        % 查找边界掩码中值为 1 的像素点的行列坐标
        [r, l] = find(bound == 1);
        for k = 1 : size(r, 1)
            % 将相邻分块的像素值合并到一个数组中
            merge = src((blocks == j) | (blocks == blocks(r(k), l(k))));
            % 计算合并后数组的极差是否小于阈值
            if (range(merge( : )) < threshold * 256)
                blocks(blocks == blocks(r(k), l(k))) = j;
            end
        end
    end

    % 根据标记重新分割
    output2 = src;
    for i = 1 : 255
        for j = 1 : 255
            % 如果当前像素点处于两个不同的分块的边界上
            if (blocks(i, j) ~= blocks(i, j + 1) || blocks(i, j) ~= blocks(i + 1, j))
                output2(i, j) = 255; % 设置边界
            end
        end
    end
    
    % show
    subplot(1, 3, 3); imshow(output2, []); title('合并结果'); 
end

