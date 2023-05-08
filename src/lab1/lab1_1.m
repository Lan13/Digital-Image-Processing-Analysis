function lab1_1(filename, tx, ty)
    src = imread(filename);
    dst = imtranslate(src, [tx, ty]);
    
    subplot(1, 2, 1); imshow(src); title('source');
    subplot(1, 2, 2); imshow(dst); title('result');
end