import numpy as np
import heapq

#actual number of boxes is this number^2
SUBDIVISIONS = 2
priority_queue = []

def process_image(image:np.ndarray, error:float, type:int)->np.ndarray:
    if is_invalid(image):
        return image
    global priority_queue
    priority_queue = []
    return compress(image, error, type)

def compress(image:np.ndarray, error:float, type:int) -> np.ndarray:
    new_image= np.zeros(shape=image.shape,dtype=np.int16)
    match type:
        case 0:
            return simple_compress(image, new_image, error)
    return image


def is_invalid(image:np.ndarray) -> bool:
    """
        checks if the array representing a image is valid 
        note: it assumes a 1x1 image is already optimal and so it returns false
    """
    if(image is None):
        return True
    size = image.shape
    if(size[0] <= 1 and size[1] <=1):
        return True
    return False

def error_percent_simple(base_image:np.ndarray, created_image:np.ndarray, 
                  x:int, y:int, width:int, height:int):
    error_sum = 0
    loop_count = 0
    for i in range(x, x + width):
        for j in range(y, y + height):
            r_err = np.abs(int(created_image[i][j][0]) - int(base_image[i][j][0]))
            g_err = np.abs(int(created_image[i][j][1]) - int(base_image[i][j][1]))
            b_err = np.abs(int(created_image[i][j][2]) - int(base_image[i][j][2]))
            error_sum += r_err + g_err+b_err
            loop_count+=3
    return (error_sum/loop_count)/255

def avg_color(image:np.ndarray, x:int, y:int, width:int, height:int):
    error_sum = 0
    loop_count = 0
    for i in range(x, x + width):
        for j in range(y, y + height):
            error_sum += int(image[i][j][0]) + int(image[i][j][1]) + int(image[i][j][2])
            loop_count+=3
    return error_sum//loop_count

def avg_color_exact(image:np.ndarray, rect:list):
    error_sum = np.zeros((image.shape[2]))
    loop_count = 0
    for i in range(rect[0], rect[0] + rect[2]):
        for j in range(rect[1], rect[1] + rect[3]):
            for k in range(image[i][j].shape[0]):
                error_sum[k] += (int)(image[i][j][k])
            loop_count += 1
    return np.array(error_sum/loop_count)

def array_write_rect(image:np.ndarray, value:np.ndarray, rect:list):
    for i in range(rect[2]):
        for j in range(rect[3]):
            image[rect[0] + i][rect[1] + j] = value


def split_range(start:int, length:int):
    """inclusive of left but not right side when looking at 2 numbers"""
    ranges = []
    for i in range(SUBDIVISIONS):
        sub_start = start + int(round((i * length) / SUBDIVISIONS))
        sub_end = start + int(round(((i + 1) * length) / SUBDIVISIONS))
        
        sub_length = sub_end - sub_start
        
        ranges.append((sub_start, sub_length))
    return ranges





def simple_compress(image:np.ndarray, new_image:np.ndarray, error:float):
    current_error = error_percent_simple(image, new_image, 0, 0, image.shape[0], image.shape[1])
    heapq.heappush(priority_queue, (- current_error, (0,0,new_image.shape[0], new_image.shape[1])))

    target_error = current_error * error

    while(current_error >= target_error and priority_queue):
        new_area = find_rect_simple()
        if((new_area[3] == 1 and new_area[2] == 1) or (new_area[3] == 0 or new_area[2] == 0)):
            continue
        x_split = split_range(new_area[0], new_area[2])
        y_split = split_range(new_area[1], new_area[3])
        for i in range(SUBDIVISIONS):
            for j in range(SUBDIVISIONS):
                start_x, width_x = x_split[j][0], x_split[j][1]
                start_y, height_y = y_split[i][0], y_split[i][1]
                new_rect = [start_x, start_y, width_x, height_y]
                if width_x <= 0 or height_y <= 0:
                    continue
                color = avg_color_exact(image, new_rect)

                array_write_rect(new_image, color, new_rect)
                new_error = error_percent_simple(image, new_image, new_rect[0], new_rect[1], new_rect[2], new_rect[3], )
                heapq.heappush(priority_queue, (-new_error, new_rect))
        
        current_error = error_percent_simple(image, new_image, 0, 0, image.shape[0], image.shape[1])
    return new_image




def find_rect_simple():
    """
        find the next area to break up in to 4 parts
    """
    item = heapq.heappop(priority_queue)
    return item[1]
    





    

        
