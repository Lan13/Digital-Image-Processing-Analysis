function lab2_1(filename, fa, fb)
    src = imread(filename);
    [row, col, ~] = size(src);
    dst = zeros(row, col);
    for y = 1 : row
        for x = 1 : col
            dst(y, x) = src(y, x) * fa + fb;
        end
    end
    
    subplot(1, 2, 1); imshow(src); title('source');
    subplot(1, 2, 2); imshow(uint8(dst)); title('result');
end