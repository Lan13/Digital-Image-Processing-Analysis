function lab5_2(filename)
    src = imread(filename);
    % 大津法获得阈值
    threshold = graythresh(src);
    % 分割图像
    output = imbinarize(src, threshold);
    % show
    subplot(1, 2, 1); imshow(src, []); title('原图像'); 
    subplot(1, 2, 2); imshow(output, []); title('分割结果'); 
end