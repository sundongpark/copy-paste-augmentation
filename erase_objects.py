import glob
import tqdm
import cv2
import numpy as np
import os

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def pad_image_to_square(img):
    height, width = img.shape
    margin = [np.abs(height - width) // 2, np.abs(height - width) // 2]

    if np.abs(height-width) % 2 != 0:
        margin[0] += 1

    if height < width:
        margin_list = [margin, [0, 0]]
    else:
        margin_list = [[0, 0], margin]
    return np.pad(img, margin_list, mode='constant')

def erase_zero(matrix):
    # Count the number of rows and columns in the matrix
    rows = len(matrix)
    cols = len(matrix[0])

    # Find the rows and columns that are all zeros
    zero_rows = [i for i in range(rows) if all(val == 0 for val in matrix[i])]
    zero_cols = [j for j in range(cols) if all(matrix[i][j] == 0 for i in range(rows))]

    # Find the rows and columns that have at least one non-zero value
    non_zero_rows = [i for i in range(rows) if i not in zero_rows]
    non_zero_cols = [j for j in range(cols) if j not in zero_cols]

    # Extract the non-zero submatrix
    submatrix = np.array([[matrix[i][j] for j in non_zero_cols] for i in non_zero_rows])

    return submatrix

PATH = '230220_filter_full'
CLASSES = ['background', 'bottle', 'can', 'chain',
               'drink-carton', 'hook', 'propeller', 'shampoo-bottle',
               'standing-bottle', 'tire', 'valve', 'wall']

classes_num = [0 for i in range(len(CLASSES))]

if __name__ == '__main__':
    for i in range(1, len(CLASSES)):
        createFolder(f'./{PATH}/object_Images/{i}')
    createFolder(f'./{PATH}/removed')
    imgs = glob.glob(f'./{PATH}/Images/*.png')
    masks = glob.glob(f'./{PATH}/Masks/*.png')
    for i in tqdm.tqdm(range(len(masks))):
        mask = cv2.imread(masks[i], cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(imgs[i], cv2.IMREAD_GRAYSCALE)
        remove = img.copy()
        for c in range(1,len(CLASSES)):
            c_img = img.copy()
            if np.any(mask == c):
                for y in range(mask.shape[0]):
                    for x in range(mask.shape[1]):
                        if mask[y][x] != c:
                            c_img[y][x] = 0
                        else:
                            remove[y][x] = 0
                c_img = pad_image_to_square(erase_zero(c_img))
                remove[y][x] = 0

                cv2.imwrite(f'./{PATH}/object_Images/{c}/{str(classes_num[c]).zfill(4)}.png', c_img)
                classes_num[c] = classes_num[c] + 1
        cv2.imwrite(f'./{PATH}/removed/{str(i).zfill(4)}.png', remove)