from PIL import Image
import numpy as np

def main():
    img = Image.open("Solid_blue.png")
    img_arr = np.array(img)

if __name__ == "__main__":
    main()