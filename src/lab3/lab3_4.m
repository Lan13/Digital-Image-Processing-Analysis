function lab3_4(filename, threshold)
    src = imread(filename);
    p_src = imnoise(src, 'salt & pepper', 0.1);
    g_src = imnoise(src, 'gaussian');
    [row, col, ~] = size(src);
    temp = imnoise(src, 'salt & pepper', 0.1);
    r_src = src;
    for i = 1 : row
        for j = 1 : col
            if (temp(i, j) ~= src(i, j))
                r_src(i, j) = uint8(rand() * 255);
            end
        end
    end
    % threshold median filter
    p_dst = filter(p_src, threshold);
    g_dst = filter(g_src, threshold);
    r_dst = filter(r_src, threshold);
    % show
    subplot(3, 3, 2); imshow(src); title('原图像');
    
    subplot(3, 3, 4); imshow(p_src); title('椒盐噪声');
    subplot(3, 3, 5); imshow(g_src); title('高斯噪声');
    subplot(3, 3, 6); imshow(r_src); title('随机噪声');
    
    subplot(3, 3, 7); imshow(p_dst); title('椒盐超限中值滤波');
    subplot(3, 3, 8); imshow(g_dst); title('高斯超限中值滤波');
    subplot(3, 3, 9); imshow(r_dst); title('随机超限中值滤波');
end

% threshold median filter
function [out] = filter(in, threshold)
    out = in;
    [row, col, ~] = size(in);
    for i = 2 : (row - 1)
        for j = 2 : (col - 1)
            temp = in(i - 1: i + 1, j - 1 : j + 1);
            median_value = median(temp(:));
            if (abs(double(in(i, j)) - double(median_value)) > threshold)
                out(i, j) = median_value;
            end
        end
    end
end