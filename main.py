# Final Version
# Vision Based Obstacle Detection

import os
import sys
from tkinter import *
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox as mb
from PIL import ImageTk, Image
import cv2
import webbrowser
import numpy as np
from matplotlib import pyplot as plt

#globals
img1 = None
img2 = None

MIN_MATCH_COUNT = 10

def donothing():
    print("Doing nothing...placeholder")

def refresh(self):
    self.destroy()
    launch_gui()

def launch_gui():
    print("Launching GUI")

    # intialize tkinter gui
    root = Tk()
    root.title("Vision Based Obstactle Detection")
    root.geometry("2200x1080")
    root.resizable(False, False)

    menubar = Menu(root)

    tools_menu = Menu(menubar, tearoff=0)
    tools_menu.add_command(label="Reset GUI", command=lambda: refresh(root))
    tools_menu.add_separator()
    tools_menu.add_command(label="Exit", command=lambda: sys.exit())

    menubar.add_cascade(label="Tools", menu=tools_menu)

    helpmenu = Menu(menubar, tearoff=0)
    helpmenu.add_command(label="How To Guide", command=lambda: webbrowser.open('https://github.com/mattmartin1/4990-final/blob/main/HowToGuide.md'))
    helpmenu.add_command(label="Github Project", command=lambda: webbrowser.open('https://github.com/mattmartin1/4990-final'))
    helpmenu.add_command(label="Gain Access", command=lambda: mb.showinfo("GitHub Access", "To gain access to the Git repository, please email marti13r@uwindsor.ca"))
    menubar.add_cascade(label="Help", menu=helpmenu)

    root.config(menu=menubar)

    #intialize frames
    frame1 = Frame(root, width=200, height=150)
    frame2 = Frame(root, width=200, height=150)
    frame3 = Frame(root, width=550, height=550)
    frame4 = Frame(root, width=250, height=250, bg = 'black')
    frame5 = Frame(root, width=250, height=250, bg = 'black')

    header = Label(root, font=("Arial", 25), text="Welcome to the Vision Based Obstactle Detection Demo")
    header.pack(side=TOP)

    #this button loads image into frame 1
    load_image_1 = Button(frame1, text="Load Image", command= lambda: load_image(frame1, 1))
    load_image_1.pack(side=BOTTOM)

    #this button loads image into frame 2
    load_image_2 = Button(frame2, text="Load Image", command= lambda: load_image(frame2, 2))
    load_image_2.pack(side=BOTTOM)

    #click this button to detect obstacles by finding sample matches
    sift_button = Button(frame3, text="Obstacle Detection From Sample", command= lambda: sift_images(frame3))
    sift_button.pack(side=BOTTOM)

    # click this button to detect obstacles using a crop and compare
    crop_button = Button(frame3, text="Cropped Obstacle Detection", command=lambda: crop_images(frame3))
    crop_button.pack(side=BOTTOM)

    # click this button to detect obstacles with trained model
    train_button = Button(frame3, text="Trained Obstacle Detection", command=lambda: train_images(frame3))
    train_button.pack(side=BOTTOM)

    #load blank image into sample frame
    frame1.pack()
    frame1.place(x=25, y=225)
    img = Image.open("1.png")
    img_resized = img.resize((250, 250))
    img_new = ImageTk.PhotoImage(img_resized)
    label = Label(frame1, image=img_new)
    label.pack()

    # load blank image into source frame
    frame2.pack()
    frame2.place(x=25, y=550)
    img2 = Image.open("2.png")
    img2_resized = img2.resize((250, 250))
    img2_new = ImageTk.PhotoImage(img2_resized)
    label2 = Label(frame2, image=img2_new)
    label2.pack()

    # main frame
    frame3.pack()
    frame3.place(x=300, y=60)
    img3 = Image.open("blank.jpg")
    img3_resized = img3.resize((1600, 900))
    img3_new = ImageTk.PhotoImage(img3_resized)
    label3 = Label(frame3, image=img3_new)
    label3.pack()

    # create textbox for process details
    frame4.pack()
    frame4.place(x=1925, y=225)
    launch_gui.Text1 = tk.Text(frame4)
    launch_gui.Text1.place(relx=0.03, rely=0.03, relheight=0.95, relwidth=0.95, bordermode='ignore')
    launch_gui.Text1.bind("<Key>", lambda a: "break")

    # create textbox for obstacle detection status
    frame5.pack()
    frame5.place(x=1925, y=550)
    launch_gui.Text2 = tk.Text(frame5)
    launch_gui.Text2.place(relx=0.03, rely=0.03, relheight=0.95, relwidth=0.95, bordermode='ignore')
    launch_gui.Text2.bind("<Key>", lambda a: "break")

    mb.showinfo("Welcome", "Please select [Help] --> [How to Guide] to learn how to use this demo")
    #run the gui
    root.mainloop()

#this functions lets a user load an image into the attached frame
def load_image(frame,frame_number):
    launch_gui.Text1.delete('1.0', END)
    launch_gui.Text1.insert(END, print_frame("Loading image into GUI"))

    #clear old data from frame
    clearFrame(frame)

    #get the file path from user
    filename = filedialog.askopenfilename(initialdir="", title="Select file",
                                          filetypes=(("png files", "*.*"), ("all files", "*.*")))

    #open the new image
    image = Image.open(filename)
    image_resized = image.resize((250, 250))
    image_new = ImageTk.PhotoImage(image_resized)

    # pass back the loaded file path
    global img1
    global img2

    if frame_number == 1:
        img1 = filename
        launch_gui.Text1.delete('1.0', END)
        launch_gui.Text1.insert(END, print_frame("Loaded "+ img1))
        print("Loaded", img1)

    if frame_number == 2:
        img2 = filename
        launch_gui.Text1.delete('1.0', END)
        launch_gui.Text1.insert(END, print_frame("Loaded "+ img2))
        print("Loaded", img2)

    #attach the new image as a label
    new_label = Label(frame, image=image_new)
    new_label.pack()

    #readd the original button
    new_button = Button(frame, text="Load Image", command=lambda: load_image(frame, frame_number))
    new_button.pack(side=BOTTOM)

    frame.mainloop()

#wipe all children from a frame
def clearFrame(frame):
    for widget in frame.winfo_children():
        widget.destroy()


def crop_images(frame):
    launch_gui.Text1.delete('1.0', END)
    launch_gui.Text1.insert(END, print_frame("Cropping images and looking for obstacles"))

    global img1
    global img2

    # handler to ensure there are enough images to sift
    if img1 == None or img2 == None:
        mb.showerror("Error", "Please ensure you have 2 images loaded into the application")
        return

    crop_img1 = cv2.imread(img1)
    crop_img2 = cv2.imread(img2)

    mask1 = np.zeros(crop_img1.shape[0:2], dtype=np.uint8)
    mask2 = np.zeros(crop_img2.shape[0:2], dtype=np.uint8)

    points = np.array([[[780, 1200], [260, 1988], [1286, 2002]]])

    # method 1 smooth region
    cv2.drawContours(mask1, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.drawContours(mask2, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)

    res1 = cv2.bitwise_and(crop_img1, crop_img1, mask=mask1)
    res2 = cv2.bitwise_and(crop_img2, crop_img2, mask=mask2)

    rect = cv2.boundingRect(points)  # returns (x,y,w,h) of the rect

    cropped1 = res1[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    cropped2 = res2[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]

    # create the white background of the same size of original image
    wbg1 = np.ones_like(crop_img1, np.uint8) * 255
    wbg2 = np.ones_like(crop_img2, np.uint8) * 255

    cv2.bitwise_not(wbg1, wbg1, mask=mask1)
    cv2.bitwise_not(wbg2, wbg2, mask=mask2)

    dst1 = wbg1 + res1
    dst2 = wbg2 + res2

    img1 = cv2.imwrite("Cropped1.jpg", cropped1)
    if os.path.exists("Cropped1.jpg"):
        crop1 = "Cropped1.jpg"

    img2 = cv2.imwrite("Cropped2.jpg", cropped2)
    if os.path.exists("Cropped2.jpg"):
        crop2 = "Cropped2.jpg"

    sift_img1 = cv2.imread(crop1, 0)
    sift_img2 = cv2.imread(crop2, 0)

    # initialize sift object
    sift = cv2.xfeatures2d.SIFT_create()

    # detect sift points in each image
    keypoints_1, descriptors_1 = sift.detectAndCompute(sift_img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(sift_img2, None)

    FLANN_INDEX_KDTREE = 1

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    new_pts = cv2.perspectiveTransform(src_pts, M)

    obstacle_flag = None
    for x, y in zip(new_pts,dst_pts):
        if x[0][0] != y[0][0] or x[0][1] != y[0][1]:
            obstacle_flag = True
            break
        else:
            obstacle_flag = False

    if obstacle_flag == True:
        launch_gui.Text2.delete('1.0', END)
        launch_gui.Text2.insert(END, print_frame("SUCCESS, an obstacle was detected in the path"))
    else:
        launch_gui.Text2.delete('1.0', END)
        launch_gui.Text2.insert(END, print_frame("FAILURE, no obstacles were found in the path"))
        
    matchesMask = mask.ravel().tolist()
    h, w = sift_img1.shape
    pts = np.float32([[w/2, 0], [0, h], [w, h]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    sift_img2 = cv2.polylines(sift_img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    
    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)

    homo_img3 = cv2.drawMatches(sift_img1, keypoints_1, sift_img2, keypoints_2, good, None, **draw_params)

    # initialize a feature matcher and match points between images
    matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = matcher.match(descriptors_1, descriptors_2)

    # sort the matches
    matches = sorted(matches, key=lambda x: x.distance)

    # show first 100 matches
    matched_img = cv2.drawMatches(sift_img1, keypoints_1, sift_img2, keypoints_2, matches[:100], sift_img2, flags=2)

    # output matched image to a jpg
    cv2.imwrite("matched_images.jpg", matched_img)

    if os.path.exists("matched_images.jpg"):
        mb.showinfo("Success", "SIFT image successfully created! Loading now...")
        filename = "matched_images.jpg"
    else:
        mb.showerror("Error", "No SIFT image was created, please try again.")
        filename = "blank.jpg"

    clearFrame(frame)

    # open the new image
    image = Image.open(filename)
    image_resized = image.resize((1600, 900))
    image_new = ImageTk.PhotoImage(image_resized)

    # attach the new image as a label
    new_label = Label(frame, image=image_new)
    new_label.pack()

    # re add the original buttons
    new_sift_button = Button(frame, text="Obstacle Detection From Sample", command= lambda: sift_images(frame))
    new_sift_button.pack(side=BOTTOM)

    new_crop_button = Button(frame, text="Cropped Obstacle Detection", command=lambda: crop_images(frame))
    new_crop_button.pack(side=BOTTOM)

    new_train_button = Button(frame, text="Trained Obstacle Detection", command=lambda: train_images(frame))
    new_train_button.pack(side=BOTTOM)

    frame.mainloop()

# this method finds obstacles by matching to a sample
def sift_images(frame):
    launch_gui.Text1.delete('1.0', END)
    launch_gui.Text1.insert(END, print_frame("Looking for obstacles that match the sample"))

    global img1
    global img2

    #handler to ensure there are enough images to sift
    if img1 == None or img2 == None:
        mb.showerror("Error", "Please ensure you have 2 images loaded into the application")
        return

    #open the images as cv2 objects
    sift_img1 = cv2.imread(img1,0)
    sift_img2 = cv2.imread(img2,0)

    #initialize sift object
    sift = cv2.xfeatures2d.SIFT_create()

    #detect sift points in each image
    keypoints_1, descriptors_1 = sift.detectAndCompute(sift_img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(sift_img2, None)

    FLANN_INDEX_KDTREE = 1

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = sift_img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        sift_img2 = cv2.polylines(sift_img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        launch_gui.Text2.delete('1.0', END)
        launch_gui.Text2.insert(END, print_frame("SUCCESS, An obtacle matching the sample was detected"))
    else:
        launch_gui.Text2.delete('1.0', END)
        launch_gui.Text2.insert(END, print_frame("FAILURE, Not enough GOOD matches were found, therefore no obstacle was detected that matches the sample"))
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)

    homo_img3 = cv2.drawMatches(sift_img1, keypoints_1, sift_img2, keypoints_2, good, None, **draw_params)

    #initialize a feature matcher and match points between images
    matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = matcher.match(descriptors_1, descriptors_2)

    #sort the matches
    matches = sorted(matches, key=lambda x: x.distance)

    #show first 100 matches
    matched_img = cv2.drawMatches(sift_img1, keypoints_1, sift_img2, keypoints_2, matches[:100], sift_img2, flags=2)

    #output matched image to a jpg
    cv2.imwrite("matched_images.jpg", matched_img)

    if os.path.exists("matched_images.jpg"):
        mb.showinfo("Success", "SIFT image successfully created! Loading now...")
    else:
        mb.showerror("Error", "No SIFT image was created, please try again.")

    clearFrame(frame)

    if os.path.exists("matched_images.jpg"):
        filename = "matched_images.jpg"
    else:
        mb.showerror("Error", "No SIFT image was found, please ensure you have sifted the images")
        filename = "blank.jpg"

    # open the new image
    image = Image.open(filename)
    image_resized = image.resize((1600, 900))
    image_new = ImageTk.PhotoImage(image_resized)

    # attach the new image as a label
    new_label = Label(frame, image=image_new)
    new_label.pack()

    # re add the original buttons
    new_sift_button = Button(frame, text="Obstacle Detection From Sample", command= lambda: sift_images(frame))
    new_sift_button.pack(side=BOTTOM)

    new_crop_button = Button(frame, text="Cropped Obstacle Detection", command=lambda: crop_images(frame))
    new_crop_button.pack(side=BOTTOM)

    new_train_button = Button(frame, text="Trained Obstacle Detection", command=lambda: train_images(frame))
    new_train_button.pack(side=BOTTOM)

    frame.mainloop()

# this method find obstacles by looking for matches to the trained model provided
def train_images(frame):
    launch_gui.Text1.delete('1.0', END)
    launch_gui.Text1.insert(END, print_frame("Looking for obstacles that match the trained model"))
    print("Detecting images using trained model")

    global img2

    #handler to ensure there are enough images to sift
    if img2 == None:
        mb.showerror("Error", "Please load a sample image")
        return

    #open the images as cv2 objects
    sift_img2 = cv2.imread(img2)

    #convert to grayscale
    sift_img2_grey = cv2.cvtColor(sift_img2, cv2.COLOR_BGR2GRAY)
    sift_img2_rgb = cv2.cvtColor(sift_img2, cv2.COLOR_BGR2RGB)

    stop_data = cv2.CascadeClassifier('trainedmodelsdemo/stop_data.xml')

    found = stop_data.detectMultiScale(sift_img2_grey,
                                       minSize=(20, 20))
    amount_found = len(found)

    if amount_found != 0:
        for (x, y, width, height) in found:
            detected = cv2.rectangle(sift_img2_rgb, (x, y), (x + height, y + width), (0, 255, 0), 5)

        launch_gui.Text2.delete('1.0', END)
        launch_gui.Text2.insert(END, print_frame("SUCCESS, Obstacles were found matching the trained model"))
    else:
        launch_gui.Text2.delete('1.0', END)
        launch_gui.Text2.insert(END, print_frame("FAILURE, no obstacles matching the trained model were found"))
        mb.showerror("FAILURE", "No obstacles matching the trained model were found, please reset the GUI and try again [Tools] --> [Reset GUI]")

    clearFrame(frame)

    # open the new image
    image = Image.fromarray(detected)
    image_resized = image.resize((1600, 900))
    image_new = ImageTk.PhotoImage(image_resized)

    canvas = tk.Canvas(frame, width=1600, height=900)
    canvas.pack()
    canvas.create_image(20, 20, anchor="nw", image=image_new)

    # re add the original buttons
    new_sift_button = Button(frame, text="Obstacle Detection From Sample", command= lambda: sift_images(frame))
    new_sift_button.pack(side=BOTTOM)

    new_crop_button = Button(frame, text="Cropped Obstacle Detection", command=lambda: crop_images(frame))
    new_crop_button.pack(side=BOTTOM)

    new_train_button = Button(frame, text="Trained Obstacle Detection", command=lambda: train_images(frame))
    new_train_button.pack(side=BOTTOM)

    frame.mainloop()

def update_text(self, value):
        self.text.set(value)

def print_frame(str):
    return str+'\n'

if __name__ == '__main__':
    launch_gui()
