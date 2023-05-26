function lab4_3(filename1, filename2)
    src1 = imread(filename1);
    src2 = imread(filename2);
    
    f1 = fft2(src1); %傅里叶变换
    f2 = fft2(src2); %傅里叶变换
    
    theta1 = angle(f1); % 取相位
    theta2 = angle(f2); % 取相位
    
    i_f1 = uint8(abs(ifft2(10000 * exp(1i * theta1)))); % 进行逆变换
    i_f2 = uint8(abs(ifft2(10000 * exp(1i * theta2)))); % 进行逆变换
    
    % show
    subplot(2, 2, 1); imshow(src1, []); title('原图像1'); 
    subplot(2, 2, 2); imshow(i_f1, []); title('相位逆变换图像'); 
    subplot(2, 2, 3); imshow(src2, []); title('原图像2'); 
    subplot(2, 2, 4); imshow(i_f2, []); title('相位逆变换图像'); 
end