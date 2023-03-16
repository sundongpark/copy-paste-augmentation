import numpy as np
import cv2
import math
import tqdm
import glob
import os

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

crop_mask = np.zeros((480, 320), dtype=np.uint8)
for hh in range(480):
    for ww in range(320):
        if abs(math.atan2(ww-160, 570-hh)) <= math.radians(15) and 110 < math.dist([hh, ww],[570, 160]) < 540:
            crop_mask[hh][ww] = 1

bg_group = glob.glob('./230220_filter_full/removed/*.png')
inpainting = 'NS' # 'Gaussian', 'Telea', 'Patchmatch', 'NS'

for b in tqdm.tqdm(bg_group):
    background = cv2.imread(b, cv2.IMREAD_GRAYSCALE)
    createFolder(f'./{inpainting}')
    if inpainting == 'Gaussian':
        non_zero_background = background[background!=0]
        m, s = np.mean(non_zero_background), np.std(non_zero_background)
        for hh in range(0,480):
            for ww in range(0,320):
                if background[hh][ww] == 0 and hh > 0 and hh < 479 and ww > 0 and ww < 319 and crop_mask[hh][ww] != 0:
                    background[hh][ww] = np.clip(int(np.random.normal(m,s)), 0, 255)# randint(50, 100)

    elif inpainting == 'Telea':
        mask = np.zeros((480, 320), dtype=np.uint8)
        mask[background == 0] = 255
        background = cv2.inpaint(background, mask, 3, cv2.INPAINT_TELEA)
    elif inpainting == 'NS':
        mask = np.zeros((480, 320), dtype=np.uint8)
        mask[background == 0] = 255
        background = cv2.inpaint(background, mask, 3, cv2.INPAINT_NS)
    elif inpainting == 'Patchmatch':
        patch_size = 17
        num_iterations = 10       

        mask = np.zeros_like(background)
        mask[background == 0] = 1
        mask *= crop_mask

        background = np.copy(background)

        for i in range(num_iterations):

            # Create a copy of the output image
            output_copy = np.copy(background)
                
            # Loop over each pixel in the masked region
            for y in range(0,mask.shape[0]-patch_size,2):
                for x in range(0,mask.shape[1]-patch_size,2):
                    # If the pixel is masked, find a matching patch from the unmasked region
                    if mask[y, x] == 1:
                        # Generate random search locations
                        search_x = np.random.randint(0, background.shape[1] - patch_size)
                        search_y = np.random.randint(0, background.shape[0] - patch_size)
                        # Extract the target patch from the output image
                        target_patch = background[y:y+patch_size, x:x+patch_size]

                        for i in range(10):
                            # Compute the distance between the target patch and the patch at the search location
                            search_patch = background[search_y:search_y+patch_size, search_x:search_x+patch_size]
                            while np.any(search_patch == 0):
                                search_x = np.random.randint(0, background.shape[1] - patch_size)
                                search_y = np.random.randint(0, background.shape[0] - patch_size)
                                search_patch = background[search_y:search_y+patch_size, search_x:search_x+patch_size]
                            distance = np.sum((target_patch - search_patch)**2)
                                
                            # If this is the best match so far, update the output image
                            if i == 0 or distance < best_distance:
                                best_distance = distance
                                best_patch = search_patch
                                    
                            # Generate a new random search location
                            search_x = np.random.randint(0, background.shape[1] - patch_size)
                            search_y = np.random.randint(0, background.shape[0] - patch_size)
                            
                        # Update the output image with the best matching patch
                        output_copy[y:y+patch_size, x:x+patch_size] = best_patch
                            
            # Copy the updated output image back to the original output image
            background = np.copy(output_copy)
    background *= crop_mask
    # Save the output image
    cv2.imwrite(f'./{inpainting}/{b[-8:]}',background)

