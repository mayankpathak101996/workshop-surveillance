from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
import tkinter as tk
import cv2
from sliding_window import TrackingClr,SelectFrame
import os
finallst = []
finallst1 = []
filnme = []
Chklst = [0]
Wrklst = []
WrkstnNm = []
RoiFrame = []
frameNo=0

class ScrolledCanvas(Frame):
    def __init__(self, master, **kwargs):
        Frame.__init__(self, master, **kwargs)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.canv = Canvas(self, bd=0, highlightthickness=0)
        self.hScroll = Scrollbar(self, orient='horizontal',
                                 command=self.canv.xview)
        self.hScroll.grid(row=1, column=0, sticky='we')
        self.vScroll = Scrollbar(self, orient='vertical',
                                 command=self.canv.yview)
        self.vScroll.grid(row=0, column=1, sticky='ns')
        self.canv.grid(row=0, column=0, sticky='nsew', padx=4, pady=4)        
        self.canv.configure(xscrollcommand=self.hScroll.set,
                            yscrollcommand=self.vScroll.set)


class MyApp(Tk):
    def __init__(self):
        Tk.__init__(self)
        self.title("WorkShop")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.main = ScrolledCanvas(self)
        self.main.grid(row=0, column=0, sticky='nsew')
        self.c = self.main.canv

        self.currentImage = {}
        menubar = Menu(self)
        filemenu = Menu(menubar, tearoff=0)
        editmenu = Menu(menubar, tearoff=0)
        Trackingmenu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File Menu", menu=filemenu)
        filemenu.add_command(label="Load Video",activebackground='grey',command=self.fileChooser)
        filemenu.add_command(label="Load URL",activebackground='grey',command=self.urlLoader)
        filemenu.add_command(label="Exit",activebackground='grey',command=self.destroy)
        menubar.add_cascade(label="Edit Menu", menu=editmenu)
        editmenu.add_command(label="Add Names",activebackground='grey',command=self.Name)
        editmenu.add_command(label="Add Workstation",activebackground='grey',command=self.WrkStation)
        menubar.add_cascade(label="Tracking", menu=Trackingmenu)
        Trackingmenu.add_command(label="Start Tracking",activebackground='grey',command=self.cvtrck)
        Trackingmenu.add_command(label="Frame Stoping Second",activebackground='grey',command=self.frmNo)
        self.config(menu=menubar)
        #self.load_imgfile(file)

        self.c.bind('<ButtonPress-1>', self.on_mouse_down)
        self.c.bind('<B1-Motion>', self.on_mouse_drag)
        self.c.bind('<ButtonRelease-1>', self.on_mouse_up)
        self.c.bind('<Button-3>', self.on_right_click)
        
        
    def fileChooser(self):
        self.file=tk.filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
        filnme.append(self.file)
        self.load_imgfile()
    def WrkStation(self):
        cap = cv2.VideoCapture(filnme[0])
        if os.path.exists(filnme[0]) == True:
            Chklst[0] = "WrkStatn"
        ok , img = cap.read()
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        #img = img.convert('RGB')
        self.currentImage['data'] = img

        photo = ImageTk.PhotoImage(img)
        self.c.xview_moveto(0)
        self.c.yview_moveto(0)
        self.c.create_image(0, 0, image=photo, anchor='nw', tags='img')
        self.c.config(scrollregion=self.c.bbox('all'))
        self.currentImage['photo'] = photo

        
    def load_imgfile(self):
        print("-*-*-*-*-*-")
        img = SelectFrame(filnme[0])
        RoiFrame.append(img)
        img = Image.fromarray(img.astype('uint8'))
        #img = img.convert('RGB')
        self.currentImage['data'] = img

        photo = ImageTk.PhotoImage(img)
        self.c.xview_moveto(0)
        self.c.yview_moveto(0)
        self.c.create_image(0, 0, image=photo, anchor='nw', tags='img')
        self.c.config(scrollregion=self.c.bbox('all'))
        self.currentImage['photo'] = photo

    def on_mouse_down(self, event):        
        self.anchor = (event.widget.canvasx(event.x),
                       event.widget.canvasy(event.y))
        self.item = None

    def on_mouse_drag(self, event):        
        bbox = self.anchor + (event.widget.canvasx(event.x),
                              event.widget.canvasy(event.y))
        if self.item is None:
            self.item = event.widget.create_rectangle(bbox, outline="yellow")
        else:
            event.widget.coords(self.item, *bbox)
        return bbox
        
    def on_mouse_up(self, event):
        if self.item:
            f,g,j,k = self.on_mouse_drag(event) 
            box = tuple((int(round(v)) for v in event.widget.coords(self.item)))

            roi = self.currentImage['data'].crop(box) # region of interest
            values = roi.getdata() # <----------------------- pixel values
            if Chklst[0] == "WrkStatn":
                Wrklst.append([f,g,j,k])
            else:
                finallst.append([f,g,j,k])
        
            #print list(values)

    def on_right_click(self, event):        
        found = event.widget.find_all()
        for iid in found:
            if event.widget.type(iid) == 'rectangle':
                event.widget.delete(iid)
    #this function is used to save the names of person and workstations            
    def Name(self):
        def printtext():
            string = e.get()
            if Chklst[0] == "WrkStatn":
                WrkstnNm.append(string)
            else:
                finallst1.append(string)   
        root = Tk()
        
        root.title('Name')
        
        e = Entry(root)
        e.pack()
        e.focus_set()
        
        b1 = Button(root,text='okay',command=printtext)
        b1.pack(side='left')
        b2 = Button(root,text='exit',command=root.destroy)
        b2.pack(side='bottom')
        root.mainloop()
    
    def urlLoader(self):
        def printtextFrame2():
            global url
            string2=e2.get()
            filnme.append(string2)
        root3=Tk()
        
        root3.title('URL Loader')
        e2=Entry(root3)
        e2.pack()
        e2.focus_set()
        b5 = Button(root3,text='okay',command=printtextFrame2)
        b5.pack(side='left')
        b6 = Button(root3,text='exit',command=root3.destroy)
        
        b6.pack(side='bottom')
        self.load_imgfile()
        root3.mainloop()
    
    def frmNo(self):
        global frameNo
        def printtextFrame():
            global frameNo
            string1=e1.get()
            frameNo=string1
            
        root1=Tk()
        root1.title('Frame Stoping Second')
        e1=Entry(root1)
        e1.pack()
        e1.focus_set()
        b3 = Button(root1,text='okay',command=printtextFrame)
        b3.pack(side='left')
        b4 = Button(root1,text='exit',command=root1.destroy)
        b4.pack(side='bottom')
        root1.mainloop()
        
        
    
    #here we are calling the tracking method 
    def cvtrck(self):
        TrackingClr(finallst,finallst1,filnme[0],Wrklst,WrkstnNm,RoiFrame[0],frameNo)
app =  MyApp()
app.mainloop()
print("workstation")
print(Wrklst)
print("-*-*-*-*-*-*-*-")
print("Tshirt roi ")
print(finallst)

