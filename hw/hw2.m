img = imread("lena.bmp");
img_salt_pepper = imnoise(img, "salt & pepper", 0.05);
img_gauss = imnoise(img, "gaussian");
n = 3;
[x, y] = size(img);
img_result1 = img;
img_result2 = img;
for i = 1 : x - (n - 1)  
    for j = 1 : y - (n - 1)  
        img_result1(i + (n - 1) / 2, j + (n - 1) / 2) = median(img_salt_pepper(i : i + (n - 1), j : j + (n - 1)), 'all');
        img_result2(i + (n - 1) / 2, j + (n - 1) / 2) = median(img_gauss(i : i + (n - 1), j : j + (n - 1)), 'all');
    end  
end
subplot(2, 2, 1); imshow(img_salt_pepper);
imwrite(img_salt_pepper, "lena_salt_pepper.bmp");
subplot(2, 2, 2); imshow(img_result1);
subplot(2, 2, 3); imshow(img_gauss);
subplot(2, 2, 4); imshow(img_result2);