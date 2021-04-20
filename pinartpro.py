import cv2
import numpy as np
import os
import copy

from PIL import Image

DEBUG = True

def detect_edges(image_path):
    # Image setup
    img0 = cv2.imread(image_path)
    gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

    # Remove noise
    img = cv2.GaussianBlur(gray,(7,7),0)

    # Convolute with proper kernels
    laplacian = cv2.Laplacian(img,cv2.CV_64F)

    # Write to file for resizing
    script_dir = os.path.dirname(__file__)
    if not os.path.exists(f'{script_dir}/tmp'):
        os.makedirs(f'{script_dir}/tmp')

    cv2.imwrite(f"{script_dir}/tmp/pic.png", laplacian)
    with Image.open(f"{script_dir}/tmp/pic.png") as orig_output:
        output = orig_output.resize((50, 50))
        output_rgb = output.convert("RGB")

    # Output RGB values for debugging purposes
    if DEBUG:
        with open(f'{script_dir}/tmp/rgb_vals.txt', 'w+') as output_fp:
            debug_output = copy.deepcopy(output_rgb)

            # Search for maximum RGB value. Greyscale so RGB are all equal
            max_magnitude = 0
            for row_i in range(0, 50):
                for col_i in range(0, 50):
                    curr_magnitude = debug_output.getpixel((row_i, col_i))[0]
                    if curr_magnitude > max_magnitude:
                        max_magnitude = curr_magnitude
            
            # Modify pixels to some stronger white value based on max magnitude
            for row_i in range(0, 50):
                for col_i in range(0, 50):
                    curr_magnitude = debug_output.getpixel((row_i, col_i))[0]
                    scaled_mag = int((curr_magnitude / max_magnitude) * 255)
                    debug_output.putpixel((row_i, col_i), (scaled_mag, scaled_mag, scaled_mag))

            debug_output.save(f'{script_dir}/tmp/resized.png')

            for row_i in range(0, 50):
                for col_i in range(0, 50):
                    output_fp.write(str(output_rgb.getpixel((row_i, col_i))) + " ")
                
                output_fp.write("\n")
            
    return output_rgb

def output_motor_data(rgb_data):
    # Search for maximum RGB value. Greyscale so RGB are all equal
    max_magnitude = 0
    for row_i in range(0, 50):
        for col_i in range(0, 50):
            curr_magnitude = rgb_data.getpixel((row_i, col_i))[0]
            if curr_magnitude > max_magnitude:
                max_magnitude = curr_magnitude


    # Write each pixel's magnitude as a percentage of the maximum magnitude as binary data
    mag_array = []
    for row_i in range(0, 50):
        for col_i in range(0, 50):
            rgb_magnitude = rgb_data.getpixel((row_i, col_i))[0]
            percentage = int(float(rgb_magnitude) * 100.0 / float(max_magnitude))
            mag_array.append(percentage)
            
    byte_arr = bytes(mag_array)
    script_dir = os.path.dirname(__file__)
    if not os.path.exists(f'{script_dir}/output'):
        os.makedirs(f'{script_dir}/output/')
    
    with open(f"{script_dir}/output/motordata", "wb+") as data_fp:
        data_fp.write(byte_arr)
            

if __name__ == "__main__":
    rgb_data = detect_edges("mario.jpg")
    output_motor_data(rgb_data)