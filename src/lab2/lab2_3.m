function lab2_3(filename, left, right)
    src = imread(filename);
    
    subplot(1, 2, 1); imshow(src); title('source');
    subplot(1, 2, 2); histogram(src, 'BinLimits' ,[left, right]); title('result');
end