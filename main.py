from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import compressor

def main():
    size = 50
    img = Image.open("dark_city.png")
    img_arr = np.array(img.resize((size, size)))

    img_1 = compressor.process_image(np.copy(img_arr), 0.25, 0)
    img_2 = compressor.process_image(np.copy(img_arr), 0.1, 0)
    img_3 = compressor.process_image(np.copy(img_arr), 0.05, 0)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    axes[0, 0].imshow(img_arr)
    axes[0, 0].set_title("Image 1: Top Left")

    axes[0, 1].imshow(img_1)
    axes[0, 1].set_title("Image 2: Top Right")

    axes[1, 0].imshow(img_2)
    axes[1, 0].set_title("Image 3: Bottom Left")

    axes[1, 1].imshow(img_3)
    axes[1, 1].set_title("Image 4: Bottom Right")

    for ax in axes.flat:
        ax.axis('off')  

    plt.tight_layout()  
    plt.show()   

if __name__ == "__main__":
    main()