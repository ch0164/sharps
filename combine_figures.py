import matplotlib.pylab as plt
from PIL import Image
import os

def main():
    fig, ax = plt.subplots(4, 3, figsize=(30, 40))
    i, j = 0, 0
    for image_name in os.listdir("image_list"):
        image = Image.open(f"image_list/{image_name}", "r")
        ax[i, j].imshow(image)
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])
        ax[i, j].set_title("Test")

        j += 1
        if j == 3:
            i += 1
            j = 0

    fig.tight_layout()
    fig.show()

if __name__ == "__main__":
    main()