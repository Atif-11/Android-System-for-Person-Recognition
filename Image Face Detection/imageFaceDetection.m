orig_img = imread("Atif's pfp.png");
[Rows, Cols] = size(orig_img);
if Rows>330
    orig_img=imresize(orig_img,[320 NaN]);
end
fd = vision.CascadeObjectDetector();
loc = step(fd, orig_img);

detected  = insertShape(orig_img, "rectangle", loc);

figure; 
imshow(detected);
title("Muhammad Atif")

