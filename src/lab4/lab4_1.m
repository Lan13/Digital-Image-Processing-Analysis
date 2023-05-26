function lab4_1(filename1, filename2)
    src1 = imread(filename1);
    src2 = imread(filename2);
    
    f1 = fft2(src1); %傅里叶变换
    f2 = fft2(src2); %傅里叶变换
    
    f_shift1 = fftshift(f1); % 移动低频
    f_shift2 = fftshift(f2); % 移动低频
    
    f_scale1 = log(abs(f_shift1 + 1)); %取幅度并进行缩放
    f_scale2 = log(abs(f_shift2 + 1)); %取幅度并进行缩放
    
    % show
    subplot(2, 2, 1); imshow(src1, []); title('原图像1');
    subplot(2, 2, 2); imshow(f_scale1, []); title('幅度变换图像'); 
    subplot(2, 2, 3); imshow(src2, []); title('原图像2'); 
    subplot(2, 2, 4); imshow(f_scale2, []); title('幅度变换图像'); 
end