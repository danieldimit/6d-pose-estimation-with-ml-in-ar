from tkinter import *
from tkinter.filedialog import askopenfilename
import sys, os

event2canvas = lambda e, c: (c.canvasx(e.x), c.canvasy(e.y))

canvas = None
img_on_canvas = None
file_counter = 0
files = []

def start_window(images_folder):
    global canvas, file_counter, files, img_on_canvas
    root = Tk()

    files = sorted(os.listdir(images_folder))
    cwd = os.path.join(os.getcwd(), images_folder)

    #setting up a tkinter canvas with scrollbars
    frame = Frame(root, bd=2, relief=SUNKEN)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    xscroll = Scrollbar(frame, orient=HORIZONTAL)
    xscroll.grid(row=1, column=0, sticky=E+W)
    yscroll = Scrollbar(frame)
    yscroll.grid(row=0, column=1, sticky=N+S)
    canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
    canvas.grid(row=0, column=0, sticky=N+S+E+W)
    xscroll.config(command=canvas.xview)
    yscroll.config(command=canvas.yview)
    frame.pack(fill=BOTH,expand=1)

    def next_picture(event):
        global file_counter, files, canvas, img_on_canvas
        file_counter += 1;
        print(os.path.join(cwd, files[file_counter]))
        img = PhotoImage(file="./rgb/0013.png")
        canvas.itemconfig(img_on_canvas,image = img)

    #function to be called when mouse is clicked
    def printcoords(event):
        #outputting x and y coords to console
        cx, cy = event2canvas(event, canvas)
        print ("(%d, %d) / (%d, %d)" % (event.x,event.y,cx,cy))
  
    #adding the image
    #print("opening %s" % f)
    img = PhotoImage(file=os.path.join(cwd, files[file_counter]))
    img_on_canvas = canvas.create_image(0,0,image=img,anchor="nw")
    canvas.config(scrollregion=canvas.bbox(ALL))

    #mouseclick event
    canvas.bind("<ButtonRelease-3>",next_picture)
    canvas.bind("<ButtonRelease-1>",printcoords)

    root.mainloop()
    
    

if __name__ == "__main__":
    if len(sys.argv) == 2:
        images_folder = sys.argv[1]
        start_window(images_folder)
    else:
        print('Usage:')
        print(' python manual_label.py images_folder')
