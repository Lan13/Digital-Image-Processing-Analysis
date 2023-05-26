function lab4_6(filename1, filename2, threshold, n)
    src1 = imread(filename1);
    src2 = imread(filename2);
    
    % add noise
    pepper_src1 = imnoise(src1, 'salt & pepper', 0.1);
    gauss_src1 = imnoise(src1, 'gaussian');
    pepper_src2 = imnoise(src2, 'salt & pepper', 0.1);
    gauss_src2 = imnoise(src2, 'gaussian');

    % show
    figure();
    subplot(3, 3, 1); imshow(src1, []); title('原图像1'); 
    subplot(3, 3, 2); imshow(pepper_src1, []); title('椒盐噪声图像1'); 
    subplot(3, 3, 3); imshow(gauss_src1, []); title('高斯噪声图像1'); 
    subplot(3, 3, 4); imshow(ILPF(pepper_src1, threshold), []); title('理想低通滤波器椒盐噪声图像'); 
    subplot(3, 3, 5); imshow(BLPF(pepper_src1, threshold, n), []); title('巴特沃斯低通滤波器椒盐噪声图像');
    subplot(3, 3, 6); imshow(ELPF(pepper_src1, threshold, n), []); title('高斯低通滤波器椒盐噪声图像');
    subplot(3, 3, 7); imshow(ILPF(gauss_src1, threshold), []); title('理想低通滤波器高斯噪声图像'); 
    subplot(3, 3, 8); imshow(BLPF(gauss_src1, threshold, n), []); title('巴特沃斯低通滤波器高斯噪声图像');
    subplot(3, 3, 9); imshow(ELPF(gauss_src1, threshold, n), []); title('高斯低通滤波器高斯噪声图像');
    
    figure();
    subplot(3, 3, 1); imshow(src2, []); title('原图像2'); 
    subplot(3, 3, 2); imshow(pepper_src2, []); title('椒盐噪声图像2'); 
    subplot(3, 3, 3); imshow(gauss_src2, []); title('高斯噪声图像2'); 
    subplot(3, 3, 4); imshow(ILPF(pepper_src2, threshold), []); title('理想低通滤波器椒盐噪声图像'); 
    subplot(3, 3, 5); imshow(BLPF(pepper_src2, threshold, n), []); title('巴特沃斯低通滤波器椒盐噪声图像');
    subplot(3, 3, 6); imshow(ELPF(pepper_src2, threshold, n), []); title('高斯低通滤波器椒盐噪声图像');
    subplot(3, 3, 7); imshow(ILPF(gauss_src2, threshold), []); title('理想低通滤波器高斯噪声图像'); 
    subplot(3, 3, 8); imshow(BLPF(gauss_src2, threshold, n), []); title('巴特沃斯低通滤波器高斯噪声图像');
    subplot(3, 3, 9); imshow(ELPF(gauss_src2, threshold, n), []); title('高斯低通滤波器高斯噪声图像');
end

function output = ILPF(input, threshold)
    [r, c, ~] = size(input);
    shift_f = fftshift(fft2(input)); %傅里叶变换并移动低频
    [u, v] = meshgrid(- c / 2 : c / 2 - 1, - r / 2 : r / 2 - 1); % 频率坐标
    dist = hypot(u, v); % 与原点的距离
    h = (dist <= threshold); % 传递函数 h
    g = shift_f.* h; % 滤波后的图像
    output = abs(ifft2(ifftshift(g))); % 傅里叶逆变换
end

function output = BLPF(input, threshold, n)
    [r, c, ~] = size(input);
    shift_f = fftshift(fft2(input)); %傅里叶变换并移动低频
    [u, v] = meshgrid(- c / 2 : c / 2 - 1, - r / 2 : r / 2 - 1); % 频率坐标
    dist = hypot(u, v); % 与原点的距离
    h = 1./ (1 + ((dist./ threshold).^ (2 * n))); % 传递函数 h
    g = shift_f.* h; % 滤波后的图像
    output = abs(ifft2(ifftshift(g))); % 傅里叶逆变换
end

function output = ELPF(input, threshold, n)
    [r, c, ~] = size(input);
    shift_f = fftshift(fft2(input)); %傅里叶变换并移动低频
    [u, v] = meshgrid(- c / 2 : c / 2 - 1, - r / 2 : r / 2 - 1); % 频率坐标
    dist = hypot(u, v); % 与原点的距离
    h = exp(- (dist./ threshold).^ n); % 传递函数 h
    g = shift_f.* h; % 滤波后的图像
    output = abs(ifft2(ifftshift(g))); % 傅里叶逆变换
end