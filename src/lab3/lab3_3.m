function lab3_3(filename)
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
    % median filter
    p_dst = medfilt2(p_src);
    g_dst = medfilt2(g_src);
    r_dst = medfilt2(r_src);
    % show
    subplot(3, 3, 2); imshow(src); title('原图像');
    
    subplot(3, 3, 4); imshow(p_src); title('椒盐噪声');
    subplot(3, 3, 5); imshow(g_src); title('高斯噪声');
    subplot(3, 3, 6); imshow(r_src); title('随机噪声');
    
    subplot(3, 3, 7); imshow(p_dst); title('椒盐中值滤波');
    subplot(3, 3, 8); imshow(g_dst); title('高斯中值滤波');
    subplot(3, 3, 9); imshow(r_dst); title('随机中值滤波');
end