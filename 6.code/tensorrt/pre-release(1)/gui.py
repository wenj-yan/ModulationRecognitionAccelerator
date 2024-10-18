
from inference import Modulation_Recognizer,Spectrogram,SAMPLE_RATE
import tkinter as tk 
import sys
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np 
import threading
import time

# sorry,should have been in inference.py I got lazy
CNN = False # if using CNN model, set to True, otherwise set to False

class Modulation_Recognizer_GUI(Modulation_Recognizer):
    def __init__(self, model_fname, model_addroot, name_network="LModCNNResNetRelu", dynamic_input_shp=[128,2], output_shp=7, spec_len=1024, ip_address='localhost', destination_port=5421,canvas=None,gui=False,onnx_fname = "res.onnx"):
        super().__init__(model_fname, model_addroot, name_network, dynamic_input_shp, output_shp, 
                         spec_len, ip_address, destination_port,gui=True,onnx_fname=onnx_fname)
        self.canvas = canvas
        self.plot_thread = threading.Thread(target=self.update_plot,args=(self.canvas,))
    def start(self):
        self.RUN_FLAG = True
        self.data_thread.start()
        self.plot_thread.start()
    def update_plot(self,canvas):    # try:
        print(f'update_plot:{self.RUN_FLAG}')
        while True:
            if self.RUN_FLAG:
                if not self.queue.empty():
                    print("update_plot")
                    trunk = self.queue.get()
                    print(trunk.shape)
                    self.spec.update(trunk[0,:,:])
                    if CNN:
                        #1,128,2 
                        p_trunk = trunk.transpose(0,2,1)
                        prediction = self.predict(p_trunk
                                                  )
                    else:
                        prediction = self.predict(trunk)
                    self.plot_draw(canvas,trunk,prediction,type="fft")
            else:
                time.sleep(0.1)
    def plot_draw(self, canvas, trunk, prediction, type="constellation"):
        
        axs = [canva.figure.axes[0] for canva in canvas]
        axs[0].clear()
        axs[0].plot(trunk[0, :, 0])
        axs[0].plot(trunk[0, :, 1])
        axs[0].set_title(f'{prediction} time domain signal')
        
        
        self.spec.spectro_draw_v2(axs[1])
        axs[1].set_title(f'{prediction} spectrogram')
        
        axs[2].clear()
        if type == "constellation":
            I = np.array(trunk[0, :, 0]).flatten()
            Q = np.array(trunk[0, :, 1]).flatten()
            axs[2].hist2d(I, Q, bins=150, range=[[-1.5, 1.5], [-1.5, 1.5]], cmap=plt.cm.binary)
            axs[2].set_aspect('equal', 'box')
            axs[2].set_title(f'{self.stat}')
        elif type == "fft":
            I = np.array(trunk[0, :, 0]).flatten()
            Q = np.array(trunk[0, :, 1]).flatten()
            fft = np.fft.fft(trunk[0, :, 0] + 1j * trunk[0, :, 1])
            fft_freq = np.fft.fftfreq(len(fft), 1 / SAMPLE_RATE)
            axs[2].plot(fft_freq, np.abs(fft))
            axs[2].set_title(f'{prediction} FFT')
            
        axs[3].clear()
        axs[3].plot(self.time)
        
        axs[3].plot(self.average_time)
        axs[3].legend(['time','average'])
        axs[3].set_title('Time Spent for each prediction')
        for canva in canvas:
            canva.draw()
    
MODEL_ADDROOT = "./jupyter/log"
MODEL_NAME = "model-AugMod-LModCNNResNetRelu-trained128.h5"
class TextRedirector(object):
    def __init__(self, widget):
        self.widget = widget

    def write(self, str):
        self.widget.insert(tk.END, str)
        self.widget.see(tk.END)
    def flush(self):
        pass
class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.grid()
        

        self.create_widgets()
        
    def create_widgets(self):
        # 创建一个新的matplotlib图形
        fig1 = Figure(figsize=(5, 4), dpi=100)
        ax = fig1.add_subplot(111)
        fig2 = Figure(figsize=(5, 4), dpi=100)
        ax = fig2.add_subplot(111)
        fig3 = Figure(figsize=(5, 4), dpi=100)
        ax = fig3.add_subplot(111)
        
        fig4 = Figure(figsize=(5, 4), dpi=100)
        ax = fig4.add_subplot(111)
        
        self.title = tk.Label(self, text="Modulation Recognizer", font=("Helvetica", 32))
        self.title.grid(row=0, column=0, columnspan=6)
        
        self.canvas1 = FigureCanvasTkAgg(fig1, master=self)
        self.canvas1.get_tk_widget().grid(row=1, column=0,columnspan=2)
        self.canvas2 = FigureCanvasTkAgg(fig2, master=self)
        self.canvas2.get_tk_widget().grid(row=1, column=2,columnspan=2)
        self.canvas3 = FigureCanvasTkAgg(fig3, master=self)
        self.canvas3.get_tk_widget().grid(row=1, column=4,columnspan=2)
        self.canvas4 = FigureCanvasTkAgg(fig4, master=self)
        self.canvas4.get_tk_widget().grid(row=4, column=3,columnspan=3)
        self.canvas = [self.canvas1,self.canvas2,self.canvas3,self.canvas4]
        
        self.mod_rec = Modulation_Recognizer_GUI(model_fname=MODEL_NAME,
                                                destination_port=2024,
                                                model_addroot=MODEL_ADDROOT,
                                                name_network="LModCNNResNetRelu",
                                                dynamic_input_shp=[128,2],canvas=self.canvas,
                                                onnx_fname="res.onnx")

        # 创建开始/结束按钮
        self.start_button = tk.Button(self, text="START", command=self.start,font=("Helvetica", 16))
        self.start_button.grid(row=2, column=0,columnspan=2)
        self.stop_button = tk.Button(self, text="PAUSE", command=self.stop,font = ("Helvetica", 16))
        self.stop_button.grid(row=2, column=2,columnspan=2)
        self.quit_button = tk.Button(self, text="EXIT", command=self.master.destroy,font = ("Helvetica", 16))    
        self.quit_button.grid(row=2, column=4,columnspan=2)
        # 创建edit
        self.spec_len_label = tk.Label(self, text="spec_len",font=("Helvetica", 16))
        self.spec_len_label.grid(row=3, column=0)
        self.spec_len = tk.IntVar()
        self.spec_len.trace_add("write", self.on_spec_len_change)
        self.input_entry_s = tk.Scale(self, from_=128, to=2048, orient=tk.HORIZONTAL, variable=self.spec_len, resolution=128,font=("Helvetica", 14))
        self.input_entry_s.grid(row=3, column=1)
        self.spec_len.set(1024)
        
        self.nfft_label = tk.Label(self, text="nfft",font=("Helvetica", 16))
        self.nfft_label.grid(row=3, column=2)
        self.nfft = tk.IntVar()
        self.nfft.trace_add("write", self.on_nfft_change)
        self.input_entry_f = tk.Scale(self, from_=32, to=1024, orient=tk.HORIZONTAL, variable=self.nfft, resolution=32,font=("Helvetica", 14)  )
        self.input_entry_f.grid(row=3, column=3)
        self.nfft.set(128)
        
        self.port_label = tk.Label(self, text="Port",font=("Helvetica", 16))
        self.port_label.grid(row=3, column=4)
        self.port = tk.IntVar()
        self.port_entry = tk.Entry(self, textvariable=self.port,font=("Helvetica", 14))
        self.port.trace_add("write", self.on_port_change)
        self.port_entry.grid(row=3, column=5)
        self.port.set(2024)
        
        
        
        # 创建一个Text控件
        self.text_box = tk.Text(self)
        self.text_box.grid(row=4, column=0, columnspan=3)

        # 重定向stdout
        sys.stdout = TextRedirector(self.text_box)
        # 启动进程
        self.mod_rec.start()
        
        
    def on_port_change(self, *args):
        self.mod_rec.destination_port = int(self.port.get())
    def on_spec_len_change(self, *args):
        self.mod_rec.spec_len = int(self.spec_len.get())
        self.mod_rec.spec.set_spec_len(int(self.spec_len.get()))
        
    def on_nfft_change(self, *args):
        self.mod_rec.nfft = int(self.nfft.get())
        self.mod_rec.spec.set_nfft(int(self.nfft.get()))
        
    def start(self):
        print("start")
        self.mod_rec.RUN_FLAG = True
            

    def stop(self):
        self.mod_rec.RUN_FLAG = False

root = tk.Tk()
app = Application(master=root)
app.mainloop()

