function lab1_4(filename1, filename2)
    src1 = imread(filename1);
    src2 = imread(filename2);
    subplot(1, 2, 1); imshow(src1); title('source 1');
    subplot(1, 2, 2); imshow(src2); title('source 2');
    
    [x, y] = ginput(8);
    coord1 = [x(1) y(1); x(2) y(2); x(3) y(3); x(4) y(4)];
    coord2 = [x(5) y(5); x(6) y(6); x(7) y(7); x(8) y(8)];
    trans = fitgeotrans(coord1, coord2, 'projective');
    
    dst1 = imwarp(src1, trans, 'nearest');
    dst2 = imwarp(src1, trans, 'bilinear');
    subplot(1, 2, 1); imshow(dst1); title('nearest');
    subplot(1, 2, 2); imshow(dst2); title('bilinear');
end