from tkinter import *
from tkinter import ttk
import tensorflow as tf
import numpy as np
from tensorflow import keras
from PIL import Image, ImageDraw, ImageGrab
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

class Paint(object):

    DEFAULT_PEN_SIZE = 25.0
    DEFAULT_COLOR = 'black'
    FILE_NAME = 'number_to_guess.png'
    FILE_RESIZED = 'number_resized.png'
    FILE_GREYSCALE = 'number_greyscale.png'

    def __init__(self, nn):
        self.root = Tk()
        self.nn = nn

        #Setting the scene with buttons, boundries made of grey labels and creating canvas
        self.guess_button = ttk.Button(self.root, text='Guess', command=self.guess)
        self.guess_button.grid(row=0, column=0)

        self.erase_button = ttk.Button(self.root, text='Erase', command=self.erase)
        self.erase_button.grid(row=1, column=0)

        self.guess_label = Label(self.root, text='Guess: ', anchor='w', width=6, bg="white")
        self.guess_label.grid(row=2, column=0)

        #Visual boundries for the canvas
        self.filler1 = Label(self.root, text="", width=45, bg="grey")
        self.filler1.grid(row=0, column=1)
        self.filler2 = Label(self.root, text="", width=45, bg="grey")
        self.filler2.grid(row=1, column=1)
        self.filler3 = Label(self.root, text="", width=45, bg="grey")
        self.filler3.grid(row=2, column=1)
        self.filler4 = Label(self.root, text="", width=9, height=25, bg="grey")
        self.filler4.grid(row=3, column=0)

        self.canvas = Canvas(self.root, bg='white', width=400, height=400)
        self.canvas.grid(row=3, column=1, columnspan=5)

        #self.guess_label.pack()
        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.DEFAULT_PEN_SIZE
        self.color = self.DEFAULT_COLOR
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset)

    def guess(self):
        self.get_canvas_picture()
        self.resize_picture()
        self.get_greyscale_picture()
        image_data = self.get_pixel_values()
        prediction = self.nn.model.predict([image_data])
        predicted_number = np.argmax(prediction)
        self.guess_label['text'] = "Guess: " + str(predicted_number.item())


    def get_canvas_picture(self):
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        ImageGrab.grab().crop((x, y, x1, y1)).save(self.FILE_NAME)

    def resize_picture(self):
        img = Image.open(self.FILE_NAME)
        size = 28, 28
        img.thumbnail(size, Image.ANTIALIAS)
        img.save(self.FILE_RESIZED, "png")

    def rgb2gray(self, rgb):
        # Formula to convert to Greyscale from RGB
        # Y' = 0.2989 R + 0.5870 G + 0.1140 B

        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    def get_greyscale_picture(self):
        img = Image.open('number_resized.png').convert('L')
        img.save(self.FILE_GREYSCALE)

    def get_pixel_values(self):
        img = cv2.imread(self.FILE_GREYSCALE, 0)
        data = np.asarray(img)

        #revert greyscale values, the ones used in mnist are: 0 - white, 255-black. In greyscale
        #it is reverted
        numpy_array = np.empty(shape=(28,28), dtype=np.float16)

        for x in range(0, data.shape[0]):
            for y in range(0, data.shape[1]):
                number = float(data[x,y])
                numpy_array[x,y] = (255-number)/255

        numpy_array = np.expand_dims(numpy_array, axis=0)
        return numpy_array

    def erase(self):
        self.canvas.delete("all")

    def paint(self, event):
        self.line_width = self.line_width
        paint_color = self.color
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

class NeuralNetwork(object):

    def __init__(self):
        self.data = keras.datasets.mnist
        self.model = keras.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(10, activation="softmax")
        ])
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    def trainNN(self):
        (training_images, training_labels), (test_images, test_labels) = self.data.load_data()
        training_images = training_images / 255.0
        test_images = test_images / 255.0

        self.model.fit(training_images, training_labels, epochs=30)

        test_loss, test_acc = self.model.evaluate(test_images, test_labels)
        print(test_acc)

if __name__ == '__main__':
    nn = NeuralNetwork()
    nn.trainNN()
    Paint(nn)