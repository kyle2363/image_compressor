import numpy as np

#actual number of boxes is this number^2
SUBDIVISIONS = 2

def process_image(image:np.array, error:float, type:int)->np.array:
    if is_invalid(image):
        return None
    return compress(image, error, type)

def compress(image:np.array, error:float, type:int)->np.array:
    new_image= np.zeros(shape=image.shape,dtype=np.int16)
    match type:
        case 0:
            return simple_compress(image, new_image, error)


def is_invalid(image:np.array) -> bool:
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

def error_percent_simple(base_image:np.array, created_image:np.array, 
                  x:int, y:int, width:int, height:int):
    error_sum = 0
    loop_count = 0
    for i in range(x, x + width):
        for j in range(y, y + height):
            base_pixel_color = (int)(base_image[i][j][0] + base_image[i][j][1] + base_image[i][j][2])
            created_pixel_color = (int)(created_image[i][j][0] + created_image[i][j][1] + created_image[i][j][2])
            error_sum += np.abs(created_pixel_color - base_pixel_color)
            loop_count+=3
    return error_sum/loop_count

def avg_color(image:np.array, x:int, y:int, width:int, height:int):
    error_sum = 0
    loop_count = 0
    for i in range(x, x + width):
        for j in range(y, y + height):
            error_sum += image[i][j][0]/3 + image[i][j][1]/3 + image[i][j][2]/3
            loop_count+=1
    return error_sum//loop_count

def avg_color_exact(image:np.array, x:int, y:int, width:int, height:int):
    error_sum = np.zeros((image.shape[2]))
    loop_count = 0
    for i in range(x, x + width):
        for j in range(y, y + height):
            for k in range(image[i][j].shape[0]):
                error_sum[k] += (int)(image[i][j][k])
            loop_count += 1
    return np.array(error_sum/loop_count)

def array_write_rect(image:np.array, value:np.array, rect:tuple[int, int, int, int]):
    for i in range(rect[2]):
        for j in range(rect[3]):
            image[rect[0] + i][rect[1] + j] = value



def simple_compress(image:np.array, new_image:np.array, error:float):
    current_error = error_percent_simple(image, new_image, 0, 0, image.shape[0], image.shape[1])
    while(current_error >= error):
        new_area = find_rec_simple(image, new_image, 0, 0, new_image.shape[0], new_image.shape[1])
        for i in range(SUBDIVISIONS):
            for j in range(SUBDIVISIONS):
                color = avg_color_exact(image, new_area[0] + (int)(i*new_area[2]/SUBDIVISIONS), new_area[1] + (int)(j*new_area[3]/SUBDIVISIONS), (int)(new_area[2]/SUBDIVISIONS), (int)(new_area[3]/SUBDIVISIONS))
                array_write_rect(new_image, color, [new_area[0] + (int)(i*new_area[2]/SUBDIVISIONS), new_area[1] + (int)(j*new_area[3]/SUBDIVISIONS), (int)(new_area[2]/SUBDIVISIONS), (int)(new_area[3]/SUBDIVISIONS)])
        current_error = error_percent_simple(image, new_image, 0, 0, image.shape[0], image.shape[1])
        if((new_area[3] == 1 and new_area[2] == 1) or (new_area[3] == 0 and new_area[2] == 0)):
            return new_image
    return new_image


def find_rec_simple(image:np.array, new_image:np.array, x:int, y:int, width:int, height:int):
    """
        find the next area to break up in to 4 parts
    """
    if(width <= 1 and height <= 1):
        return (x, y, width, height)
    total_avg_color = avg_color(new_image, x, y, width, height)
    per_division_color = []
    per_division_error = []
    for i in range(SUBDIVISIONS):
        for j in range(SUBDIVISIONS):
            per_division_color.append(avg_color(new_image, x + (int)(width/SUBDIVISIONS)*j, 
                                                y + (int)(height/SUBDIVISIONS)*i, (int)(width/SUBDIVISIONS), (int)(height/SUBDIVISIONS)))
            per_division_error.append(error_percent_simple(image, new_image, x + (int)(width/SUBDIVISIONS)*j, y + (int)(height/SUBDIVISIONS)*i, (int)(width/SUBDIVISIONS), (int)(height/SUBDIVISIONS)))
    if(np.unique(per_division_color).shape[0] == 1 or (width <= 1 and height <= 1)):
        return (x, y, width, height)
    else:
        max_value = max(per_division_error)
        max_index = per_division_error.index(max_value)
        #possible future change to still check other SUBDIVISIONS - 1 boxes before going a level beyond this
        return find_rec_simple(image, new_image, x + (int)(width/SUBDIVISIONS)*((int)(max_index % SUBDIVISIONS)), y + (height//SUBDIVISIONS)*(max_index // SUBDIVISIONS), (int)(width/SUBDIVISIONS), (int)(height/SUBDIVISIONS))





    

        
