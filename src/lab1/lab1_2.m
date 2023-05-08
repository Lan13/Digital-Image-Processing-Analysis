function lab1_2(filename, angle)
    src = imread(filename);
    dst1 = imrotate(src, angle, 'nearest', 'crop');
    dst2 = imrotate(src, angle, 'bilinear', 'crop');
    
    subplot(1, 3, 1); imshow(src); title('source');
    subplot(1, 3, 2); imshow(dst1); title('nearest');
    subplot(1, 3, 3); imshow(dst2); title('bilinear');
end