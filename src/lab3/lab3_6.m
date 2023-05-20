function lab3_6(filename)
    src = imread(filename);
    
    % edge detection
    roberts = edge(src, 'Roberts');
    sobel = edge(src, 'Sobel');
    prewitt = edge(src, 'Prewitt');
    laplace1 = imfilter(src, [0 1 0; 1 -4 1; 0 1 0]);
    laplace2 = imfilter(src, [0 1 0; 1 -4 1; 0 1 0]);
    canny = edge(src, 'Canny');
    
    % show
    subplot(3, 3, 2); imshow(src); title('原图像');
    
    subplot(3, 3, 4); imshow(roberts); title('Roberts');
    subplot(3, 3, 5); imshow(sobel); title('Sobel');
    subplot(3, 3, 6); imshow(prewitt); title('Prewitt');
    
    subplot(3, 3, 7); imshow(laplace1); title('Laplace1');
    subplot(3, 3, 8); imshow(laplace2); title('Laplace2');
    subplot(3, 3, 9); imshow(canny); title('Canny');
end