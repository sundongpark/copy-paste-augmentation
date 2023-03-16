import cv2
import numpy as np
import tqdm
import imutils
import math
import glob
import os
import cv2
import numpy as np
import math

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def distance(x1, y1, x2, y2):
    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    return np.linalg.norm(p2 - p1)

def angle(x1, y1, x2, y2):
    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    v = p2 - p1
    return np.arctan2(v[1], v[0]) * 180 / np.pi

def find_starting_point(image, j):
    for i in range(image.shape[0]):
        if image[i][j] != 0:
            return i
    return None

def draw_line_with_angle(img, start_point, length, angle, color, thickness=1):
    """각도에 따라 선분 그리기"""
    end_point = (int(start_point[0] - length * math.cos(math.radians(angle))),
                 int(start_point[1] - length * math.sin(math.radians(angle))))
    cv2.line(img, start_point, end_point, color, thickness)

CLASSES = ['bottle', 'can', 'chain',
               'drink-carton', 'hook', 'propeller', 'shampoo-bottle',
               'standing-bottle', 'tire', 'valve']

SHADOW = True
RS = False
inpainting = 'patchmatch' # 'wo', 'gaussian', 'telea', 'patchmatch'
PATH = f'./full_{inpainting}'
if SHADOW:
    PATH += '_shadow'
if RS:
    PATH += '_rs'
print(PATH)

if __name__ == '__main__':
    createFolder(f'{PATH}/Images')
    createFolder(f'{PATH}/Masks')
    object_group = dict()
    if inpainting == 'patchmatch':
        bg_group = glob.glob('./PatchMatch/*.png')
    elif inpainting == 'gaussian':
        bg_group = glob.glob('./Gaussian/*.png')
    elif inpainting == 'telea':
        bg_group = glob.glob('./Telea/*.png')
    else:
        bg_group = glob.glob('./230220_filter_full/removed/*.png')
    camera_pos = (570,160)

    for i, c in enumerate(CLASSES):
        object_group[c] = glob.glob(f'./230220_filter_full/object_Images/{i+1}/*.png')

    crop_mask = np.zeros((480, 320), dtype=np.uint8)
    for hh in range(480):
        for ww in range(320):
            if abs(math.atan2(ww-160, 570-hh)) <= math.radians(15) and 110 < math.dist([hh, ww],[570, 160]) < 540:
                crop_mask[hh][ww] = 1

    for i in tqdm.tqdm(range(938*2,938*3)):
        background = cv2.imread(np.random.choice(bg_group), cv2.IMREAD_GRAYSCALE)
        mask_background = np.zeros((480,320), dtype=np.uint8)

        objects_num = np.random.randint(1,4)

        '''
        for c in np.random.choice(range(len(CLASSES)), objects_num, p=[0.07619002771,
                                                                        0.1119930731,
                                                                        0.02401489648,
                                                                        0.1897216484,
                                                                        0.1194860858,
                                                                        0.0473576057,
                                                                        0.07006693558,
                                                                        0.07593035765,
                                                                        0.03157960506,
                                                                        0.2536597645],replace=False):
        '''
        for c in np.random.choice(range(len(CLASSES)), objects_num, replace=False):                         
            object_name = CLASSES[c]
            obj_img = cv2.imread(np.random.choice(object_group[object_name]), cv2.IMREAD_GRAYSCALE)

            rand_x = np.random.randint(40, 200)
            rand_y = np.random.randint(0, 400)
            if RS:
                M = cv2.getRotationMatrix2D((obj_img.shape[0]//2, obj_img.shape[1]//2),np.random.rand()*30-15, np.random.rand()*0.1+0.95) # -15~15, 0.95~1.05
                obj_img = cv2.warpAffine(obj_img, M,(obj_img.shape[0], obj_img.shape[1]))
            if SHADOW:
                shadow_mask = np.zeros((480,320), dtype=np.uint8)
                for y in range(obj_img.shape[0]):
                    for x in range(obj_img.shape[1]):
                        if obj_img[y,x] > 0 and rand_y+y < 480 and rand_x+x < 320:
                            object_pos = [rand_y-obj_img.shape[0]//2,rand_x+obj_img.shape[1]//2]
                            shadow_dist = distance(object_pos[1],object_pos[0],camera_pos[1],camera_pos[0])/4
                            shadow_angle = angle(object_pos[1],object_pos[0],camera_pos[1],camera_pos[0])
                            draw_line_with_angle(shadow_mask, [rand_x+x,rand_y+y], shadow_dist, shadow_angle, 1, thickness=1)
                for y in range(background.shape[0]):
                    for x in range(background.shape[1]):
                        if shadow_mask[y,x]:
                            background[y,x] = background[y,x] * np.clip(np.random.normal(0.7,0.1), 0, 1)

            for y in range(obj_img.shape[0]):
                for x in range(obj_img.shape[1]):
                    if obj_img[y,x] > 0 and rand_y+y < 480 and rand_x+x < 320 and background[rand_y+y, rand_x+x] > 0:
                        background[rand_y+y, rand_x+x] = obj_img[y,x]
                        mask_background[rand_y+y, rand_x+x] = c + 1

        cv2.imwrite(f'{PATH}/Images/{str(i).zfill(4)}.png', background,)
        cv2.imwrite(f'{PATH}/Masks/{str(i).zfill(4)}.png', mask_background)