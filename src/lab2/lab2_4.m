function lab2_4(filename)
    src = imread(filename);
    eq = histeq(src);
    dist = (0 : 1 : 255);
    eq2 = histeq(src, normpdf(dist, 127, 32));
    
    subplot(3, 2, 1); imshow(src); title('原图像');
    subplot(3, 2, 2); histogram(src); title('直方图');
    
    subplot(3, 2, 3); imshow(eq); title('增强图像');
    subplot(3, 2, 4); histogram(eq); title('直方图');
    
    subplot(3, 2, 5); imshow(eq2); title('规定图像');
    subplot(3, 2, 6); histogram(eq2); title('直方图');
end