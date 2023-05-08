function lab1_3(filename, c, d)
    src = imread(filename);
    [row, col, ~] = size(src);
    dst1 = imresize(src, [row * d, col * c], 'nearest');
    dst2 = imresize(src, [row * d, col * c], 'bilinear');
    
    subplot(1, 3, 1); imshow(src); title('source');
    subplot(1, 3, 2); imshow(dst1); title('nearest');
    subplot(1, 3, 3); imshow(dst2); title('bilinear');
end