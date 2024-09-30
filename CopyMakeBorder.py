import cv2
import numpy as np

def main():
    
    with open("README.md", mode="r") as f:
        line = f.readline()
        
    print("LINE:", line)
    
    tokens = line.split()
    
    for i, t in enumerate(tokens):
        print("TOKEN", i, "-->", t)
    
    # regular division vs. integer division
    value = 5 / 2
    ivalue = 5 // 2
    print("VALUES:", value, ivalue)
    
    image = cv2.imread("test.png")
    padded = cv2.copyMakeBorder(image, top=50, bottom=300, left=4, right=71,
                                borderType=cv2.BORDER_CONSTANT, value=0)
    
    cv2.imshow("ORIGINAL", image)
    cv2.imshow("PADDED", padded)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()