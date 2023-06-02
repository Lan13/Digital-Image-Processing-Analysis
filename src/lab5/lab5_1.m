function lab5_1(filename)
    src = imread(filename);
    blurred_kernel = fspecial('motion', 30, 45); % 模糊核
    motion_blurred = imfilter(im2double(src), blurred_kernel, 'conv', 'circular'); % 运动模糊图像
    guass_blurred = imnoise(motion_blurred, 'gaussian', 0, 0.00001);
    
    % 逆滤波
    motion_inverse = deconvwnr(motion_blurred, blurred_kernel);
    gauss_inverse = deconvwnr(guass_blurred, blurred_kernel);
    
    % 维纳滤波
    motion_wiener = deconvwnr(motion_blurred, blurred_kernel); % 没有噪声
    gauss_wiener = deconvwnr(guass_blurred, blurred_kernel, 0, 0.00001 );
    
    % show
    subplot(3, 3, 1); imshow(src, []); title('原图像'); 
    subplot(3, 3, 2); imshow(motion_blurred, []); title('运动模糊图像'); 
    subplot(3, 3, 3); imshow(guass_blurred, []); title('高斯噪声模糊图像'); 
    
    subplot(3, 3, 5); imshow(motion_inverse, []); title('运动模糊逆滤波图像'); 
    subplot(3, 3, 6); imshow(gauss_inverse, []); title('高斯噪声模糊逆滤波图像'); 
    
    subplot(3, 3, 8); imshow(motion_wiener, []); title('运动模糊维纳滤波图像'); 
    subplot(3, 3, 9); imshow(gauss_wiener, []); title('高斯噪声模糊维纳滤波图像'); 
end