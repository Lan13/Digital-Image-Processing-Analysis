function lab4_2(filename1, filename2)
    src1 = imread(filename1);
    src2 = imread(filename2);
    
    f1 = fft2(src1); %傅里叶变换
    f2 = fft2(src2); %傅里叶变换
    
    i_f1 = uint8(ifft2(abs(f1))); % 取幅度并进行逆变换
    i_f2 = uint8(ifft2(abs(f2))); % 取幅度并进行逆变换
    
    % show
    subplot(2, 2, 1); imshow(src1, []); title('原图像1'); 
    subplot(2, 2, 2); imshow(i_f1, []); title('幅度逆变换图像');
    subplot(2, 2, 3); imshow(src2, []); title('原图像2'); 
    subplot(2, 2, 4); imshow(i_f2, []); title('幅度逆变换图像'); 
end