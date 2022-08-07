from PyQt5 import QtWidgets
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtGui import QIcon, QImage, QPixmap
from Task1 import Ui_ImageFilter
import numpy as np
import cv2
from PIL import Image, ImageFilter


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):#-------Init function
        super(MainWindow, self).__init__()
        self.ui = Ui_ImageFilter()
        self.ui.setupUi(self)
        #______________________________IMPORTANT INSTANCE VARIABLES__________________________________________________________#
        self.The_Image=[]
        self.file_name=[]
        self.The_Filtered_Image = []
        #___________________________________________CONNECTING BUTTONS WITH THEIR FUNCTIONS_______________________________________#
        self.ui.Upload_The_Image.clicked.connect(lambda: self.Upload_An_Image())
        self.ui.Choose_Filter.activated.connect(lambda: self.Choose_A_Filter())
        self.ui.Apply_Equalization.clicked.connect(lambda: self.Apply_Equalization())
        
#_____________________________________________BUTTTONS FUNCTIONS_______________________________________________________#
    
    def make_histogram(self,img):
        """ Take an image and create a historgram from it's luma values """
        y_vals = img[:,:,0].flatten()
        histogram = np.zeros(256, dtype=int)
        for y_index in range(y_vals.size):
            histogram[y_vals[y_index]] += 1
        return histogram
#____________________________________________________________________________________________________#  
    def Get_Frequency_Domain(self,The_Photo,The_axes,The_Canvas):
        img = img = cv2.imread(The_Photo, cv2.IMREAD_GRAYSCALE) # read the image as gray scale
        fft = np.fft.fft2(img)# fast fourier transform of the image
        # shift the zero-frequncy component to the center of the spectrum
        fft_shift = np.fft.fftshift(fft)
        # save image of the image in the fourier domain.
        magnitude_spectrum = magnitude_spectrum = 20*np.log(np.abs(fft_shift))#magnitude in db
        The_axes.imshow(magnitude_spectrum, cmap = 'gray')
        The_Canvas.draw()#show the freq domain in ui space
#____________________________________________________________________________________________________#        
    def showHistogramoriginal(self):
        """showing the histogram of the original image"""
        self.ui.axes_of_original_histo.clear()
        self.originalImageTuple = Image.open(self.file_name).convert('YCbCr')#convert image to ycbcr 
        self.W, self.H =self.originalImageTuple.size #resolution of image 
        self.originalImage = np.array(self.originalImageTuple)
        self.originalHistogram = self.make_histogram(self.originalImage)
        x_axis = np.arange(256)
        self.ui.axes_of_original_histo.bar(x_axis, self.originalHistogram, color="green")
        self.ui.canvas_of_original_histo.draw()
#____________________________________________________________________________________________________#   
    def Show_Images_and_Histo_and_freq(self,image,name_of_image,Label_To_show_image,axes,canvas,showhisto):
        image = QImage(name_of_image)
        Label_To_show_image.setPixmap(QPixmap(image))#show the image in the original image space in ui
        self.Get_Frequency_Domain(name_of_image, axes, canvas)
        showhisto  #self.showHistogramoriginal()
#____________________________________________________________________________________________________#
    def Upload_An_Image(self):#--->>LOAD THE image<<
        self.file_name, _ = QFileDialog.getOpenFileName(
            self, "Open file", ".", "Image Files (*.png *.jpg *.bmp)"
        )
        if not self.file_name:
            return
        self.Show_Images_and_Histo_and_freq(self.The_Image,self.file_name,self.ui.Original_image,self.ui.axes_of_original_freq,self.ui.canvas_of_original_freq,self.showHistogramoriginal())
#____________________________________________________________________________________________________#
    def Choose_A_Filter(self):
        if self.ui.Choose_Filter.currentIndex() == 0:#sharpen filter
            image = cv2.imread(self.file_name)
            # Print error message if image is null
            if image is None:
                print('Could not read image')
            kernel = np.array([[0.0, -1.0, 0.0],
                               [-1.0, 5.0, -1.0],
                               [0.0, -1.0, 0.0]])
            kernel = kernel / (np.sum(kernel) if np.sum(kernel) != 0 else 1)
            sharp_img = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)#depth =-1 will give the output image depth as same as the input image
            cv2.imwrite('filter.jpg', sharp_img)
            self.Show_Images_and_Histo_and_freq(self.The_Filtered_Image, 'filter.jpg', self.ui.Filtered_Image,self.ui.axes_of_filtered_freq, self.ui.canvas_of_filtered_freq,self.showHistogramFilter())

    #---------------------------------------------------------------------------------------------
        if self.ui.Choose_Filter.currentIndex() == 1:#bilateral Filter (smoothing the image and noise removal)
            image = cv2.imread(self.file_name)
            bilateral_filter = cv2.bilateralFilter(src=image, d=9, sigmaColor=75, sigmaSpace=75)#d=diameter of pixel,sicolor=Value of sigma in the color space,sigmaSpace=Value of sigma  in the coordinate space
            cv2.imwrite('filter.jpg', bilateral_filter)
            self.Show_Images_and_Histo_and_freq(self.The_Filtered_Image, 'filter.jpg', self.ui.Filtered_Image,self.ui.axes_of_filtered_freq, self.ui.canvas_of_filtered_freq,self.showHistogramFilter())
    #------------------------------------------------------------------------------------------------
        if self.ui.Choose_Filter.currentIndex() == 2:#median Blur filter (This operation processes the edges while removing the noise.)
            image = cv2.imread(self.file_name)
            figure_size = 9 #kernal size
            new_image = cv2.medianBlur(image, figure_size)
            cv2.imwrite('filter.jpg', new_image)
            self.Show_Images_and_Histo_and_freq(self.The_Filtered_Image, 'filter.jpg', self.ui.Filtered_Image,self.ui.axes_of_filtered_freq, self.ui.canvas_of_filtered_freq,self.showHistogramFilter())
    #---------------------------------------------------------------------------------------------
        if self.ui.Choose_Filter.currentIndex() == 3:#laplacian filter(This determines if a change in adjacent pixel values is from an edge or continuous progression.)
            image = cv2.imread(self.file_name)
            l_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])# edge detector
            l_kernel = l_kernel / (np.sum(l_kernel) if np.sum(l_kernel) != 0 else 1)
            temp = cv2.filter2D(image, -1, l_kernel)
            cv2.imwrite('filter.jpg', temp)
            self.Show_Images_and_Histo_and_freq(self.The_Filtered_Image, 'filter.jpg', self.ui.Filtered_Image,self.ui.axes_of_filtered_freq, self.ui.canvas_of_filtered_freq,self.showHistogramFilter())
    #----------------------------------------------------------------------------------------------
        if self.ui.Choose_Filter.currentIndex() == 4:#highpass filter
            self.fft_img_filtered_OUT = []
            image = cv2.imread(self.file_name)
            img = image/float(2 ** 8)
            TFcircleIN = self.draw_cicle(shape=img.shape[:2], diamiter=50)
            TFcircleOUT1= np.zeros(img.shape[:2])+1
            TFcircleOUT=TFcircleOUT1-TFcircleIN
            TFcircleINblur = cv2.GaussianBlur(TFcircleIN, (19,19), 0)
            TFcircleOUTblur = cv2.GaussianBlur(TFcircleOUT, (19,19), 0)           
            fft_img = np.zeros_like(img, dtype=complex)
            for ichannel in range(fft_img.shape[2]):
                fft_img[:, :, ichannel] = np.fft.fftshift(np.fft.fft2(img[:, :, ichannel]))
            
            for ichannel in range(fft_img.shape[2]):
                fft_img_channel = fft_img[:, :, ichannel]
                ## circle OUT
                temp = self.filter_circle(TFcircleOUTblur, fft_img_channel)
                self.fft_img_filtered_OUT.append(temp)
            self.fft_img_filtered_OUT = np.array(self.fft_img_filtered_OUT)
            self.fft_img_filtered_OUT = np.transpose(self.fft_img_filtered_OUT, (1, 2, 0))
            print(self.fft_img_filtered_OUT)
            img_reco_filtered_OUT = self.inv_FFT_all_channel(self.fft_img_filtered_OUT)
            cv2.imwrite('filter.jpg', np.abs(img_reco_filtered_OUT) * float(2 ** 8))
            self.Show_Images_and_Histo_and_freq(self.The_Filtered_Image, 'filter.jpg', self.ui.Filtered_Image,self.ui.axes_of_filtered_freq, self.ui.canvas_of_filtered_freq,self.showHistogramFilter())
    #----------------------------------------------------------------------------------------------
        if self.ui.Choose_Filter.currentIndex() == 5:#lowpass filter
            self.fft_img_filtered_IN = []
            image = cv2.imread(self.file_name)
            img = image / float(2 ** 8)
            TFcircleIN = self.draw_cicle(shape=img.shape[:2], diamiter=200)
            TFcircleOUT1= np.zeros(img.shape[:2])+1
            TFcircleOUT=TFcircleOUT1-TFcircleIN
            TFcircleINblur = cv2.GaussianBlur(TFcircleIN, (19,19), 0)
            fft_img = np.zeros_like(img, dtype=complex)
            for ichannel in range(fft_img.shape[2]):
                fft_img[:, :, ichannel] = np.fft.fftshift(np.fft.fft2(img[:, :, ichannel]))
            if isinstance(self.fft_img_filtered_IN,(np.ndarray)):
                self.fft_img_filtered_IN=self.fft_img_filtered_IN.tolist()
            for ichannel in range(fft_img.shape[2]):
                fft_img_channel = fft_img[:, :, ichannel]
                ## circle IN
                temp = self.filter_circle(TFcircleINblur, fft_img_channel)
                self.fft_img_filtered_IN.append(temp)
            self.fft_img_filtered_IN = np.array(self.fft_img_filtered_IN)
            self.fft_img_filtered_IN = np.transpose(self.fft_img_filtered_IN, (1, 2, 0))
            img_reco_filtered_IN = self.inv_FFT_all_channel(self.fft_img_filtered_IN)
            cv2.imwrite('filter.jpg', np.abs(img_reco_filtered_IN) * float(2 ** 8))
            self.Show_Images_and_Histo_and_freq(self.The_Filtered_Image, 'filter.jpg', self.ui.Filtered_Image,self.ui.axes_of_filtered_freq, self.ui.canvas_of_filtered_freq,self.showHistogramFilter())
#____________________________________________________________________________________________________#
    def make_cumsum(self, histogram):
        """ Create an array that represents the cumulative sum of the histogram """
        cumsum = np.zeros(256, dtype=int)
        cumsum[0] = histogram[0]
        for i in range(1, 256):
            cumsum[i] = cumsum[i-1] + histogram[i]
        return cumsum
#____________________________________________________________________________________________________#
    def make_mapping(self, cumsum):
        """ Create a mapping s.t. each old colour value is mapped to a new
            one between 0 and 255. Mapping is created using:
            - M(i) = max(0, round((grey_levels*cumsum(i))/(h*w))-1)
            where g_levels is the number of grey levels in the image """
        mapping = np.zeros(256, dtype=int)
        lumaLevels = 256
        for i in range(lumaLevels):
            mapping[i] = max(0, round((lumaLevels*cumsum[i])/(self.H*self.W))-1)
        return mapping
#____________________________________________________________________________________________________#
    def apply_mapping(self, img, mapping):
        """ Apply the mapping to our image """
        new_image = img.copy()
        new_image[:,:,0] = list(map(lambda a : mapping[a], img[:,:,0]))
        return new_image
#____________________________________________________________________________________________________#
    def showHistogramFilter(self):
        """showing the histogram of the original image"""
        self.ui.axes_of_filtered_histo.clear()
        self.filterImageTuple = Image.open('filter.jpg').convert('YCbCr')
        self.W, self.H =self.filterImageTuple.size
        self.filterImage = np.array(self.filterImageTuple)
        self.filterHistogram = self.make_histogram(self.filterImage)
        self.x_axis = np.arange(256)
        self.ui.axes_of_filtered_histo.bar(self.x_axis, self.filterHistogram, color="orange")
        self.ui.canvas_of_filtered_histo.draw()
#____________________________________________________________________________________________________#
    def showEqualizedHistogramOriginal(self):
        """showing the histogram of the original image"""
        self.ui.axes_of_original_histo.clear()
        self.originalImageTuple = Image.open(self.file_name).convert('YCbCr')
        self.W, self.H =self.originalImageTuple.size
        self.originalImage = np.array(self.originalImageTuple)
        self.originalHistogram = self.make_histogram(self.originalImage)
        self.originalCumsum = self.make_cumsum(self.originalHistogram)
        self.originalmapping = self.make_mapping(self.originalCumsum)
        self.neworiginalImage = self.apply_mapping(self.originalImage, self.originalmapping)
        self.eqOriginal = Image.fromarray(np.uint8(self.neworiginalImage), "YCbCr")
        self.eqOriginal.save('eqOriginal.jpg')
        self.x_axis = np.arange(256)
        self.ui.axes_of_original_histo.bar(self.x_axis, self.make_histogram(self.neworiginalImage), color="green")
        self.ui.canvas_of_original_histo.draw()
#____________________________________________________________________________________________________#
    def showEqualizedHistogramFilter(self):
        """showing the histogram of the original image"""
        self.ui.axes_of_filtered_histo.clear()
        self.filterImageTuple = Image.open('filter.jpg').convert('YCbCr')
        self.W, self.H =self.filterImageTuple.size
        self.filterImage = np.array(self.filterImageTuple)
        self.filterHistogram = self.make_histogram(self.filterImage)
        self.filterCumsum = self.make_cumsum(self.filterHistogram)
        self.filtermapping = self.make_mapping(self.filterCumsum)
        self.newfilterImage = self.apply_mapping(self.filterImage, self.filtermapping)
        self.eqFilter = Image.fromarray(np.uint8(self.newfilterImage), "YCbCr")
        self.eqFilter.save('eqfilter.jpg')
        self.x_axis = np.arange(256)
        self.ui.axes_of_filtered_histo.bar(self.x_axis, self.make_histogram(self.newfilterImage), color="orange")
        self.ui.canvas_of_filtered_histo.draw()
#____________________________________________________________________________________________________#     
    def Apply_Equalization(self):
        self.showEqualizedHistogramOriginal()
        self.showEqualizedHistogramFilter()
        self.The_Image = QImage('eqOriginal.jpg')
        self.ui.Original_image.setPixmap(QPixmap(self.The_Image))
        self.Get_Frequency_Domain('eqOriginal.jpg',self.ui.axes_of_original_freq,self.ui.canvas_of_original_freq)
        self.The_Filtered_Image = QImage('eqfilter.jpg')
        self.ui.Filtered_Image.setPixmap(QPixmap(self.The_Filtered_Image))
        self.Get_Frequency_Domain('eqfilter.jpg', self.ui.axes_of_filtered_freq, self.ui.canvas_of_filtered_freq)
#____________________________________________________________________________________________________#
    def draw_cicle(self,shape, diamiter):
        assert len(shape) == 2
   
        radius = diamiter//2
        mask = np.zeros(shape)
        cy = mask.shape[0] // 2
        cx = mask.shape[1] // 2
        cv2.circle(mask, (cx,cy), radius, (1,1,1), -1)[0]
        return (mask)
#____________________________________________________________________________________________________#
    def inv_FFT_all_channel(self,fft_img):
        img_reco = []
        for ichannel in range(fft_img.shape[2]):
            img_reco.append(np.fft.ifft2(np.fft.ifftshift(fft_img[:, :, ichannel])))
        img_reco = np.array(img_reco)
        img_reco = np.transpose(img_reco, (1, 2, 0))
        return (img_reco)
#____________________________________________________________________________________________________#
    def filter_circle(self,TFcircleIN, fft_img_channel):
        temp = np.zeros(fft_img_channel.shape[:2], dtype=complex)
        temp = np.multiply(fft_img_channel,TFcircleIN)
        return (temp)

    def close_app(self):
        sys.exit()
#---------------------------------END OF MAINWINDOW CLASS---------------------------------------------#
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())