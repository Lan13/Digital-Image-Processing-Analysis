function lab4_7(filename1, filename2, threshold, n)
    src1 = imread(filename1);
    src2 = imread(filename2);

    % show
    subplot(2, 4, 1); imshow(src1, []); title('原图像1'); 
    subplot(2, 4, 2); imshow(IHPF(src1, threshold), []); title('理想高通滤波器图像'); 
    subplot(2, 4, 3); imshow(BHPF(src1, threshold, n), []); title('巴特沃斯高通滤波器图像');
    subplot(2, 4, 4); imshow(EHPF(src1, threshold, n), []); title('高斯高通滤波器图像');
    subplot(2, 4, 5); imshow(src2, []); title('原图像2'); 
    subplot(2, 4, 6); imshow(IHPF(src2, threshold), []); title('理想高通滤波器图像'); 
    subplot(2, 4, 7); imshow(BHPF(src2, threshold, n), []); title('巴特沃斯高通滤波器图像');
    subplot(2, 4, 8); imshow(EHPF(src2, threshold, n), []); title('高斯高通滤波器图像');
end

function output = IHPF(input, threshold)
    [r, c, ~] = size(input);
    shift_f = fftshift(fft2(input)); %傅里叶变换并移动低频
    [u, v] = meshgrid(- c / 2 : c / 2 - 1, - r / 2 : r / 2 - 1); % 频率坐标
    dist = hypot(u, v); % 与原点的距离
    h = (dist > threshold); % 传递函数 h
    g = shift_f.* h; % 滤波后的图像
    output = abs(ifft2(ifftshift(g))); % 傅里叶逆变换
end

function output = BHPF(input, threshold, n)
    [r, c, ~] = size(input);
    shift_f = fftshift(fft2(input)); %傅里叶变换并移动低频
    [u, v] = meshgrid(- c / 2 : c / 2 - 1, - r / 2 : r / 2 - 1); % 频率坐标
    dist = hypot(u, v); % 与原点的距离
    h = 1./ (1 + ((threshold./ dist).^ (2 * n))); % 传递函数 h
    g = shift_f.* h; % 滤波后的图像
    output = abs(ifft2(ifftshift(g))); % 傅里叶逆变换
end

function output = EHPF(input, threshold, n)
    [r, c, ~] = size(input);
    shift_f = fftshift(fft2(input)); %傅里叶变换并移动低频
    [u, v] = meshgrid(- c / 2 : c / 2 - 1, - r / 2 : r / 2 - 1); % 频率坐标
    dist = hypot(u, v); % 与原点的距离
    h = exp(- (threshold./ dist).^ n); % 传递函数 h
    g = shift_f.* h; % 滤波后的图像
    output = abs(ifft2(ifftshift(g))); % 傅里叶逆变换
end