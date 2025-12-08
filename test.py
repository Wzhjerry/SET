import cv2
from skimage.segmentation import mark_boundaries, slic
from datasets import format_sub_sp

img = cv2.imread('./debug/img2.jpg')
img = cv2.resize(img, (256, 256))

sp = slic(img, n_segments=1024, compactness=10, start_label=0, max_num_iter=20)
save = mark_boundaries(image=img, label_img=sp)

sp_num = 1024
index, sub_sp = format_sub_sp(img, sp, sp_num, sp_num // 4)
index2, sub_sp2 = format_sub_sp(img, sub_sp, sp_num // 4, sp_num // 16)
index3, sub_sp3 = format_sub_sp(img, sub_sp2, sp_num // 16, sp_num // 64)
index4, sub_sp4 = format_sub_sp(img, sub_sp3, sp_num // 64, sp_num // 256)

save2 = mark_boundaries(image=img, label_img=sub_sp)
save3 = mark_boundaries(image=img, label_img=sub_sp2)
save4 = mark_boundaries(image=img, label_img=sub_sp3)
save5 = mark_boundaries(image=img, label_img=sub_sp4)

cv2.imwrite('./debug/result.png', save * 255)
cv2.imwrite('./debug/result2.png', save2 * 255)
cv2.imwrite('./debug/result3.png', save3 * 255)
cv2.imwrite('./debug/result4.png', save4 * 255)
cv2.imwrite('./debug/result5.png', save5 * 255)
