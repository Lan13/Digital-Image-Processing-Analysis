function lab4_8(filename, threshold, n, fa, fb)
    src = imread(filename);

    % show
    subplot(2, 4, 1); imshow(src, []); title('原图像'); 
    subplot(2, 4, 2); imshow(histeq(uint8(IHPF(src, threshold, fa, fb))), []); title('理想高通滤波器-直方图均衡化图像'); 
    subplot(2, 4, 3); imshow(histeq(uint8(BHPF(src, threshold, n, fa, fb))), []); title('巴特沃斯高通滤波器-直方图均衡化图像');
    subplot(2, 4, 4); imshow(histeq(uint8(EHPF(src, threshold, n, fa, fb))), []); title('高斯高通滤波器-直方图均衡化图像');
    
    subplot(2, 4, 5); imshow(src, []); title('原图像'); 
    subplot(2, 4, 6); imshow(IHPF(histeq(src), threshold, fa, fb), []); title('直方图均衡化-理想高通滤波器图像'); 
    subplot(2, 4, 7); imshow(BHPF(histeq(src), threshold, n, fa, fb), []); title('直方图均衡化-巴特沃斯高通滤波器图像');
    subplot(2, 4, 8); imshow(EHPF(histeq(src), threshold, n, fa, fb), []); title('直方图均衡化-高斯高通滤波器图像');
end

function output = IHPF(input, threshold, fa, fb)
    [r, c, ~] = size(input);
    shift_f = fftshift(fft2(input)); %傅里叶变换并移动低频
    [u, v] = meshgrid(- c / 2 : c / 2 - 1, - r / 2 : r / 2 - 1); % 频率坐标
    dist = hypot(u, v); % 与原点的距离
    h = (dist > threshold); % 传递函数 h
    h = fa * h + fb; % 高频增强
    g = shift_f.* h; % 滤波后的图像
    output = abs(ifft2(ifftshift(g))); % 傅里叶逆变换
end

function output = BHPF(input, threshold, n, fa, fb)
    [r, c, ~] = size(input);
    shift_f = fftshift(fft2(input)); %傅里叶变换并移动低频
    [u, v] = meshgrid(- c / 2 : c / 2 - 1, - r / 2 : r / 2 - 1); % 频率坐标
    dist = hypot(u, v); % 与原点的距离
    h = 1./ (1 + ((threshold./ dist).^ (2 * n))); % 传递函数 h
    h = fa * h + fb; % 高频增强
    g = shift_f.* h; % 滤波后的图像
    output = abs(ifft2(ifftshift(g))); % 傅里叶逆变换
end

function output = EHPF(input, threshold, n, fa, fb)
    [r, c, ~] = size(input);
    shift_f = fftshift(fft2(input)); %傅里叶变换并移动低频
    [u, v] = meshgrid(- c / 2 : c / 2 - 1, - r / 2 : r / 2 - 1); % 频率坐标
    dist = hypot(u, v); % 与原点的距离
    h = exp(- (threshold./ dist).^ n); % 传递函数 h
    h = fa * h + fb; % 高频增强
    g = shift_f.* h; % 滤波后的图像
    output = abs(ifft2(ifftshift(g))); % 傅里叶逆变换
end