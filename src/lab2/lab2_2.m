function lab2_2(filename, x1, y1, x2, y2)
    src = imread(filename);
    [row, col, ~] = size(src);
    dst = zeros(row, col);
    for y = 1 : row
        for x = 1 : col
            pixel = src(y, x);
            if (pixel < x1)
                dst(y, x) = pixel * (y1 / x1);
            elseif (pixel <= x2)
                dst(y, x) = (pixel - x1) * ((y2 - y1) / (x2 - x1)) + y1;
            else
                dst(y, x) = (pixel - x2) * ((255 - y2) / (255 - x2)) + y2;
            end
        end
    end
    
    subplot(1, 2, 1); imshow(src); title('source');
    subplot(1, 2, 2); imshow(uint8(dst)); title('result');
end