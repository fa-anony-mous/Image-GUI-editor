from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from tkinter.filedialog import askopenfilename, asksaveasfilename
from PIL import Image, ImageTk, ImageFilter, ImageEnhance, ImageOps, ImageMath
import cv2
import numpy as np
import os

# create root window
root = Tk()
root.title("Simple Photo Editor")
root.geometry("1920x1080")
canvas2 = Canvas(root, width=1920, height=1080, bg="white",
                 relief="raised", borderwidth=2)
canvas2.place(x=10, y=250)

kernel = np.ones((10, 10), np.float32)/25


# create functions
def selected():
    global img_path, img
    img_path = filedialog.askopenfilename(
        initialdir=os.getcwd(), title="Select file", filetypes=[("PNG file", ".png"), ("jpg file", ".jpg")])
    img = Image.open(img_path)
    img.thumbnail((350, 350))
    img1 = ImageTk.PhotoImage(img)
    canvas2.create_image(300, 260, image=img1)
    canvas2.image = img1


def save():
    global img_path, imgg, img1
    ext = img_path.split(".")[-1]
    file = asksaveasfilename(defaultextension=f".{ext}", filetypes=[(
        "All Files", "."), ("PNG file", ".png"), ("jpg file", ".jpg")])
    if file:
        if canvas2.image == img1:
            imgg.save(file)
        else:
            canvas2.image.save(file)


def reset():
    global img_path, img
    img = Image.open("reset.png")
    img.thumbnail((500, 500))
    img1 = ImageTk.PhotoImage(img)
    canvas2.create_image(300, 260, image=img1)
    canvas2.image = img1


def blur(event):
    global img_path, img1, imgg
    for m in range(0, v1.get()+1):
        img = Image.open(img_path)
        img.thumbnail((350, 350))
        imgg = img.filter(ImageFilter.GaussianBlur(radius=m*0.5))
        img1 = ImageTk.PhotoImage(imgg)
        canvas2.create_image(300, 260, image=img1)
        canvas2.image = img1


def brightness(event):
    global img_path, img2, img3
    for m in range(0, v2.get()+1):
        img = Image.open(img_path)
        img.thumbnail((350, 350))
        img2 = ImageEnhance.Brightness(img)
        img3 = img2.enhance(m*0.5)
        img4 = ImageTk.PhotoImage(img3)
        canvas2.create_image(300, 260, image=img4)
        canvas2.image = img4


def contrast(event):
    global img_path, img4, img5
    for m in range(0, v3.get()+1):
        img = Image.open(img_path)
        img.thumbnail((350, 350))
        img4 = ImageEnhance.Contrast(img)
        img5 = img4.enhance(m*0.5)
        img6 = ImageTk.PhotoImage(img5)
        canvas2.create_image(300, 260, image=img6)
        canvas2.image = img6


def rotate_image(event):
    global img_path, img6, img7
    img = Image.open(img_path)
    img.thumbnail((350, 350))
    img6 = img.rotate(int(rotate_combo.get()))
    img7 = ImageTk.PhotoImage(img6)
    canvas2.create_image(300, 260, image=img7)
    canvas2.image = img7


def flip_image(event):
    global img_path, img8, img9
    img = Image.open(img_path)
    img.thumbnail((350, 350))
    if flip_combo.get() == "Horizontal flip":
        img8 = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif flip_combo.get() == "Vertical flip":
        img8 = img.transpose(Image.FLIP_TOP_BOTTOM)
    elif flip_combo.get() == "Original":
        img8 = img
    img9 = ImageTk.PhotoImage(img8)
    canvas2.create_image(300, 260, image=img9)
    canvas2.image = img9


def image_border(event):
    # add image border
    global img_path, img10, img11
    img = Image.open(img_path)
    img.thumbnail((350, 350))
    img10 = ImageOps.expand(img, border=int(border_combo.get()))
    img11 = ImageTk.PhotoImage(img10)
    canvas2.create_image(300, 260, image=img11)
    canvas2.image = img11


img1 = None
img3 = None
img5 = None
img7 = None
img9 = None
img11 = None


def save():
    global img_path, imgg, img1, img2, img3, img4, img5, img6, img7, img8, img9, img10, img11
    # file=None
    ext = img_path.split(".")[-1]
    file = asksaveasfilename(defaultextension=f".{ext}", filetypes=[(
        "All Files", "."), ("PNG file", ".png"), ("jpg file", ".jpg")])
    if file:
        if canvas2.image == img1:
            imgg.save(file)
        elif canvas2.image == img3:
            img2.save(file)
        elif canvas2.image == img5:
            img4.save(file)
        elif canvas2.image == img7:
            img6.save(file)
        elif canvas2.image == img9:
            img8.save(file)
        elif canvas2.image == img11:
            img10.save(file)


def convert(event):
    global img_path, img1, img2
    img = Image.open(img_path)
    img.thumbnail((350, 350))
    if conversion_combo.get() == "Negative":
        img2 = ImageOps.invert(img)
    elif conversion_combo.get() == "Low Pass Filter":
        img2 = low_pass_filter(img)
    elif conversion_combo.get() == "High Pass Filter":
        img2 = high_pass_filter(img)
    elif conversion_combo.get() == "Band Pass Filter":
        img2 = band_pass_filter(img)
    elif conversion_combo.get() == "Band Stop Filter":
        img2 = band_stop_filter(img)
    elif conversion_combo.get() == "Mean Filter":
        img2 = mean_filter(img)
    elif conversion_combo.get() == "Median Filter":
        img2 = median_filter(img)
    elif conversion_combo.get() == "Bilateral Filter":
        img2 = bilateral_filter(img)
    elif conversion_combo.get() == "Box Filter":
        img2 = box_filter(img)
    elif conversion_combo.get() == "Laplacian Filter":
        img2 = laplacian_filter(img)
    elif conversion_combo.get() == "Prewitt Filter (X)":
        img2 = prewitt_filterx(img)
    elif conversion_combo.get() == "Prewitt Filter (Y)":
        img2 = prewitt_filtery(img)
    elif conversion_combo.get() == "Sobel Filter (X)":
        img2 = sobel_filterx(img)
    elif conversion_combo.get() == "Sobel Filter (Y)":
        img2 = sobel_filtery(img)
    img1 = ImageTk.PhotoImage(img2)
    canvas2.create_image(300, 260, image=img1)
    canvas2.image = img1


def morphological(event):
    global img_path, img1, img2
    img = Image.open(img_path)
    img.thumbnail((350, 350))
    # keep erosion, dilation, opening, closing
    if morphological_combo.get() == "Erosion":
        img2 = erosion(img)
    elif morphological_combo.get() == "Dilation":
        img2 = dilation(img)
    elif morphological_combo.get() == "Opening":
        img2 = opening(img)
    elif morphological_combo.get() == "Closing":
        img2 = closing(img)
    img1 = ImageTk.PhotoImage(img2)
    canvas2.create_image(300, 260, image=img1)
    canvas2.image = img1


def low_pass_filter(image):
    src = cv2.imread(img_path)
    DF = cv2.edgePreservingFilter(src, flags=1, sigma_s=60, sigma_r=0.4)
    Lp = cv2.filter2D(src, -1, kernel)
    Lp = src - Lp
    return Image.fromarray(Lp)


def high_pass_filter(image):
    src = cv2.imread(img_path)
    dst = cv2.GaussianBlur(src, (5, 5), cv2.BORDER_DEFAULT)
    Hp = src - dst
    return Image.fromarray(Hp)


def band_pass_filter(image):
    src = cv2.imread(img_path)
    DF = cv2.edgePreservingFilter(src, flags=1, sigma_s=60, sigma_r=0.4)
    Lp = cv2.filter2D(src, -1, kernel)
    Lp = src - Lp
    Hp = src - Lp
    Bp = src - Hp
    return Image.fromarray(Bp)


def band_stop_filter(image):
    src = cv2.imread(img_path)
    DF = cv2.edgePreservingFilter(src, flags=1, sigma_s=60, sigma_r=0.4)
    Lp = cv2.filter2D(src, -1, kernel)
    Lp = src - Lp
    Hp = src - Lp
    Bp = src - Hp
    Bs = src - Bp
    return Image.fromarray(Bs)


def mean_filter(image):
    src = cv2.imread(img_path)
    kernel = np.ones((10, 10), np.float32)/25
    dst2 = cv2.filter2D(src, -1, kernel)
    return Image.fromarray(dst2)


def median_filter(image):
    src = cv2.imread(img_path)
    median = cv2.medianBlur(src, 5)
    return Image.fromarray(median)


def bilateral_filter(image):
    src = cv2.imread(img_path)
    bilateral = cv2.bilateralFilter(src, 60, 60, 60)
    return Image.fromarray(bilateral)


def laplacian_filter(image):
    src = cv2.imread(img_path)
    laplacian = cv2.Laplacian(src, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    return Image.fromarray(laplacian)


def prewitt_filterx(image):
    src = cv2.imread(img_path)
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    img_prewittx = cv2.filter2D(src, -1, kernelx)
    return Image.fromarray(img_prewittx)


def prewitt_filtery(image):
    src = cv2.imread(img_path)
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    img_prewitty = cv2.filter2D(src, -1, kernely)
    return Image.fromarray(img_prewitty)


def sobel_filterx(image):
    src = cv2.imread(img_path)
    kernelx = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_sobelx = cv2.filter2D(src, -1, kernelx)
    return Image.fromarray(img_sobelx)


def sobel_filtery(image):
    src = cv2.imread(img_path)
    kernely = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    img_sobely = cv2.filter2D(src, -1, kernely)
    return Image.fromarray(img_sobely)


def box_filter(image):
    src = cv2.imread(img_path)
    box = cv2.boxFilter(src, -1, (5, 5))
    return Image.fromarray(box)


def erosion(image):
    src = cv2.imread(img_path)
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(src, kernel, iterations=1)
    return Image.fromarray(erosion)


def dilation(image):
    src = cv2.imread(img_path)
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(src, kernel, iterations=1)
    return Image.fromarray(dilation)


def opening(image):
    src = cv2.imread(img_path)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel)
    return Image.fromarray(opening)


def closing(image):
    src = cv2.imread(img_path)
    kernel = np.ones((100, 100), np.uint8)
    closing = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel)
    return Image.fromarray(closing)


# create labels, scales and comboboxes
blurr = Label(root, text="Blur:", font=("ariel 17 bold"), width=9, anchor='e')
blurr.place(x=15, y=8)
v1 = IntVar()
scale1 = ttk.Scale(root, from_=0, to=10, variable=v1,
                   orient=HORIZONTAL, command=blur)
scale1.place(x=150, y=10)

bright = Label(root, text="Brightness:", font=("ariel 17 bold"))
bright.place(x=8, y=50)
v2 = IntVar()
scale2 = ttk.Scale(root, from_=0, to=10, variable=v2,
                   orient=HORIZONTAL, command=brightness)
scale2.place(x=150, y=55)

cont = Label(root, text="Contrast: ", font=("ariel 17 bold"))
cont.place(x=35, y=92)
v3 = IntVar()
scale3 = ttk.Scale(root, from_=0, to=10, variable=v3,
                   orient=HORIZONTAL, command=contrast)
scale3.place(x=150, y=100)

rotate = Label(root, text="Rotate:", font=("ariel 17 bold"))
rotate.place(x=370, y=8)
values = [0, 90, 180, 270, 360]
rotate_combo = ttk.Combobox(root, values=values, font=('ariel 10 bold'))
rotate_combo.place(x=460, y=15)
rotate_combo.bind("<<ComboboxSelected>>", rotate_image)

flip = Label(root, text="Flip: ", font=("ariel 17 bold"))
flip.place(x=370, y=50)
values1 = ["Horizontal flip", "Vertical flip", "Original"]
flip_combo = ttk.Combobox(root, values=values1, font=('ariel 10 bold'))
flip_combo.place(x=460, y=57)
flip_combo.bind("<<ComboboxSelected>>", flip_image)

border = Label(root, text="Add border:", font=("ariel 17 bold"))
border.place(x=320, y=92)
values2 = [i for i in range(0, 45, 5)]
border_combo = ttk.Combobox(root, values=values2, font=("ariel 10 bold"))
border_combo.place(x=460, y=99)
border_combo.bind("<<ComboboxSelected>>", image_border)


# create labels, scales and combobox
conversion_label = Label(
    root, text="Filters:", font=("ariel 17 bold"))
conversion_label.place(x=110, y=150)

conversion_options = ["Negative", "Low Pass Filter", "High Pass Filter", "Band Pass Filter", "Band Stop Filter", "Mean Filter",
                      "Median Filter", "Bilateral Filter", "Box Filter", "Laplacian Filter", "Prewitt Filter (X)", "Prewitt Filter (Y)", "Sobel Filter (X)", "Sobel Filter (Y)"]
conversion_combo = ttk.Combobox(
    root, values=conversion_options, width=20, font=("ariel 10 bold"))
conversion_combo.place(x=250, y=155)
conversion_combo.bind("<<ComboboxSelected>>", convert)

# create labels, scales and combobox for morphological operations
morphological_label = Label(
    root, text="Select morphological operations:", font=("ariel 17 bold"))
morphological_label.place(x=810, y=150)

morphological_options = ["Erosion", "Dilation", "Opening", "Closing"]
morphological_combo = ttk.Combobox(
    root, values=morphological_options, width=20, font=("ariel 10 bold"))
morphological_combo.place(x=1260, y=155)
morphological_combo.bind("<<ComboboxSelected>>", morphological)

# create buttons
btn1 = Button(root, text="Select Image", bg='black', fg='gold',
              font=('ariel 15 bold'), relief=GROOVE, command=selected)
btn1.place(x=700, y=695)

btn2 = Button(root, text="Reset", width=12, bg='black', fg='gold',
              font=('ariel 15 bold'), relief=GROOVE, command=reset)
btn2.place(x=880, y=695)

btn3 = Button(root, text="Save", width=12, bg='black', fg='gold',
              font=('ariel 15 bold'), relief=GROOVE, command=save)
btn3.place(x=1060, y=695)

btn4 = Button(root, text="Exit", width=12, bg='black', fg='gold',
              font=('ariel 15 bold'), relief=GROOVE, command=root.destroy)
btn4.place(x=1240, y=695)

root.mainloop()
