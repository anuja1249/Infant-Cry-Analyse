#Smart Infant Cry Analyses and Detection Application
#Project proposed and leaded by Mr.Osmani (Paris-13)

#Library Importation
import os
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from Tkinter import *
from essentia.standard import *
import csv
import random
from sklearn import neighbors
from sklearn import metrics
from sklearn import model_selection
from sklearn import naive_bayes
from sklearn.neural_network import MLPClassifier
from sklearn import svm

#***********************************************************************************************************************
#---------------------Class Application (Main of the graphical interface of the application)----------------------------
#***********************************************************************************************************************
class Application(Frame):

    # ******************************************************************************************************************
    # I- Graphicale interface and Widgets setting
    # ******************************************************************************************************************

    #Initialisation and default parameters
    def __init__(self, param="initial", master=None, winType='hann', WinSize=1024, shifting=0.5, SampleRate=8100,
                 FreqCut=2000):

        self.FreqCut = int(FreqCut)
        self.winType = winType
        self.WinSize = int(WinSize)
        self.Shift = shifting
        self.hopSize = int(self.Shift * self.WinSize)
        self.SampleRate = int(SampleRate)
        self.datapath = r"/home/med/test/databases/ExtractedFeaturesg.csv"
        self.Audiopath = ""
        self.Energythreshold = {'pain':[], 'deaf': [], 'hunger': [], 'normal': [], 'asphyxia': []}
        self.AudioPool = {'pain': [], 'deaf': [], 'hunger': [], 'normal': [], 'asphyxia': []}
        self.SegAudioPool = {'pain': [], 'deaf': [], 'hunger': [], 'normal': [], 'asphyxia': []}
        self.FsegPool = {'pain': [], 'deaf': [], 'hunger': [], 'normal': [], 'asphyxia': []}
        self.PAudioSegPool = {'pain': [], 'deaf': [], 'hunger': [], 'normal': [], 'asphyxia': []}
        self.Feautures = {'pain': [], 'deaf': [], 'hunger': [], 'normal': [], 'asphyxia': []}
        self.meanenergy = {'pain': [], 'deaf': [], 'hunger': [], 'normal': [], 'asphyxia': []}
        Frame.__init__(self, master)
        master.title("Baby Cry Audio Analyse Application")
        self.pack()
        self.createWidgets()

    #Widgets Creation
    def createWidgets(self):

        #Set of parameters input and thier labels
        self.Lup = Label(self, text="Parameters", width=20)

        # Widgets for Windowing parametres input(Type)
        self.LwinType = Label(self, text="winType")
        self.contentWinType = StringVar(self)
        self.contentWinType.set("")
        self.entryWinType = Entry(self, textvariable=self.contentWinType, width=8)

        # Widgets for Windowing parametres input (Size)
        self.LWinSize = Label(self, text="WinSize")
        self.contentWinSize = StringVar(self)
        self.contentWinSize.set("")
        self.entryWinSize = Entry(self, textvariable=self.contentWinSize, width=8)

        # Widgets for shifting parameter input
        self.LShift = Label(self, text="shifting")
        self.contentShift = StringVar(self)
        self.contentShift.set("")
        self.entryShift = Entry(self, textvariable=self.contentShift, width=8)

        # Widgets for Sample rate parameter input
        self.LSamplRate = Label(self, text="SampleRate")
        self.contentSampleRate = StringVar(self)
        self.contentSampleRate.set("")
        self.entrySampleRate = Entry(self, textvariable=self.contentSampleRate, width=8)

        #Widgets for Filter frequence parameter input
        self.LFreq = Label(self, text="Filter F")
        self.contentFreq = StringVar(self)  # define variable that we will read from entry box
        self.contentFreq.set("")
        self.entryFreq = Entry(self, textvariable=self.contentFreq, width=8)

        #Widgets for path of data folder input
        self.LPath = Label(self, text="Path")
        self.contentPath = StringVar(self)  # define variable that we will read from entry box
        self.contentPath.set("")
        self.entryPath = Entry(self, textvariable=self.contentPath, width=30)

        #Widgets to display executed algorithmes
        self.LStatus = Label(self, text="Status")
        self.textStatus = Text(self, width=25, height=1)

        #Widgets for machine learning algorithme runing
        self.LMachineLearning = Label(self, text="Machine Learning", width=20)
        self.LMLAlgo = Label(self, text="Chose Algo")
        MLalgoList = ["KNN", "MLP","SVM","NB"]
        self.MLVar = StringVar()
        self.MLVar.set("Select")  # default choice
        self.MLList = OptionMenu(self, self.MLVar, *MLalgoList, command=self.getalgo)

        #set of visual text of algorithme performances
        self.Occurancy = Label(self, text="Occurance")
        self.Occutext = Text(self, width=10, height=1)

        self.Precision = Label(self, text="Precision")
        self.Precitext = Text(self, width=10, height=1)

        self.Fmesure = Label(self, text="Fmesure")
        self.Fmestext = Text(self, width=10, height=1)

        self.Rappel = Label(self, text="Rappel")
        self.Rapptext = Text(self, width=10, height=1)


        self.MatricC = Label(self, text="Matrice de Confusion", width=15)
        self.MatricCText = Text(self, width=15, height=7)


        # Set of execution Buttons Widgets
        self.ExInit = Button(self, text="Initialisation", width=12, command=lambda: self.setparam())

        self.ExAudioLoading = Button(self, text="AudioLoading", width=12, command=lambda: self.AudioLoading(self.contentPath.get()))

        self.ExSegmentation = Button(self, text="Segmentation", width=12, command=lambda: self.Segmentation(canvas, ax))

        self.ExFiltring = Button(self, text="Filtring", width=12, command=lambda: self.Filtering(canvas, ax))

        self.ExPreprocessing = Button(self, text="Preprocessing", width=12, command=lambda: self.preprocessing(canvas, ax))

        self.ExExtraction = Button(self, text="Features", width=12, command=lambda: self.Extractfeature(canvas, ax))

        self.ExDatabase = Button(self, text="Build DB", width=12, command=lambda: self.saveData())

        self.ExLearning = Button(self, text="Learning", width=12, command=lambda: self.ClassificationAlgo(canvas, ax))

        # Widgets for axes display
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        canvas = FigureCanvasTkAgg(fig, self)
        self.clearb = Button(self, text="Clear", width=10)
        self.clearb["command"] = self.clearall
        self.QUIT = Button(self, text="QUIT", fg="red", width=10)
        self.QUIT["command"] = self.quit

        # Grid of all widgets
        # parameter widgets
        rowparameter = 1
        colparameter = 0
        self.Lup.grid(row=0, column=4, pady=4,sticky=W + E)
        self.LwinType.grid(row=rowparameter, column=colparameter, sticky=W)
        self.entryWinType.grid(row=rowparameter, column=colparameter + 1, padx=4, sticky=W)
        self.LWinSize.grid(row=rowparameter, column=colparameter + 2, sticky=E)
        self.entryWinSize.grid(row=rowparameter, column=colparameter + 3, padx=4, sticky=W)
        self.LShift.grid(row=rowparameter, column=colparameter + 4, sticky=E)
        self.entryShift.grid(row=rowparameter, column=colparameter + 5, padx=4, sticky=W)
        self.LSamplRate.grid(row=rowparameter, column=colparameter + 6, sticky=E)
        self.entrySampleRate.grid(row=rowparameter, column=colparameter + 7, padx=4, sticky=W)
        self.LFreq.grid(row=rowparameter, column=colparameter + 8, padx=4, pady=0, sticky=E)
        self.entryFreq.grid(row=rowparameter, column=colparameter + 9, padx=4, pady=4, sticky=W)
        self.LPath.grid(row=rowparameter+1, column=colparameter+0, pady=8, sticky=W)
        self.entryPath.grid(row=rowparameter+1, column=colparameter + 1,columnspan=4, padx=4, sticky=W)
        self.LStatus.grid(row=3, column=0, pady=0, sticky=W)
        self.textStatus.grid(row=3, column=1, columnspan=8,padx=4, pady=0, sticky=W + E)

        #button widgets
        self.ExInit.grid(row=4, column=10,padx=4, sticky=E)
        self.ExAudioLoading.grid(row=5, column=10,padx=4, sticky=E)
        self.ExSegmentation.grid(row=6, column=10,padx=4, sticky=E)
        self.ExFiltring.grid(row=7, column=10,padx=4, sticky=N + E)
        self.ExPreprocessing.grid(row=8, column=10,padx=4, sticky=N + E)
        self.ExExtraction.grid(row=9, column=10,padx=4, sticky=N + E)
        self.ExDatabase.grid(row=10, column=10,padx=4, sticky=N + E)

        canvas.get_tk_widget().grid(row=4, column=1, columnspan=9, rowspan=7, pady=20, sticky=W + E)
        canvas.show()

        #learning part widgets
        self.LMachineLearning.grid(row=12, column=2, columnspan=5,sticky=W + E)
        self.ExLearning.grid(row=12, column=10, sticky=N + E)
        self.LMLAlgo.grid(row=12, column=0, sticky=W)
        self.MLList.grid(row=12, column=1, sticky=W)

        self.Occurancy.grid(row=13, column=0, pady=4, sticky=E)
        self.Occutext.grid(row=13, column=1,  pady=4, sticky=W)

        self.Precision.grid(row=13, column=2, pady=4, sticky=E)
        self.Precitext.grid(row=13, column=3, pady=4, sticky=W)

        self.Fmesure.grid(row=13, column=4, pady=4, sticky=E)
        self.Fmestext.grid(row=13, column=5, pady=4, sticky=W)

        self.Rappel.grid(row=13, column=6, pady=4, sticky=E)
        self.Rapptext.grid(row=13, column=7, pady=4, sticky=W)

        self.MatricC.grid(row=14, column=2, padx=10, sticky=E)
        self.MatricCText.grid(row=14, column=3, rowspan=2,columnspan=4, padx=20, sticky=W+E)

        self.clearb.grid(row=16, column=10, sticky=E + S)
        self.QUIT.grid(row=17, column=10, sticky=E + S)

    # ******************************************************************************************************************
    # II- Processing function defintion
    # ******************************************************************************************************************

    #Select the algo to use for machine learning
    def getalgo(self,value):
        self.algoselected = value # set the algo selected variable

    # Function to set parameters according to the graphical input labels
    def setparam(self):
        self.FreqCut = int(self.contentFreq.get())
        self.winType = self.contentWinType.get()
        self.WinSize = int(self.contentWinSize.get())
        self.Shift = float(self.contentShift.get())
        self.hopSize = int(self.Shift * self.WinSize)
        self.SampleRate = int(self.contentSampleRate.get())
        self.Audiopath = self.contentPath.get()
        self.textStatus.delete('1.0', END)
        self.textStatus.insert(INSERT, "Paramaters are intialised !")

    # Function to initialise used vectors for data processing
    def clearall(self):
        self.Energythreshold = {'pain':[], 'deaf': [], 'hunger': [], 'normal': [], 'asphyxia': []}
        self.AudioPool = {'pain': [], 'deaf': [], 'hunger': [], 'normal': [], 'asphyxia': []}
        self.SegAudioPool = {'pain': [], 'deaf': [], 'hunger': [], 'normal': [], 'asphyxia': []}
        self.FsegPool = {'pain': [], 'deaf': [], 'hunger': [], 'normal': [], 'asphyxia': []}
        self.PAudioSegPool = {'pain': [], 'deaf': [], 'hunger': [], 'normal': [], 'asphyxia': []}
        self.Feautures = {'pain': [], 'deaf': [], 'hunger': [], 'normal': [], 'asphyxia': []}
        self.meanenergy = {'pain': [], 'deaf': [], 'hunger': [], 'normal': [], 'asphyxia': []}



    # ******************************************************************************************************************
    # a-Loading Audio files from a folder given as input parameter
    # ******************************************************************************************************************
    def AudioLoading(self, path):

        # instantiation
        self.setparam()

        #Set status as Loading audio..
        self.Occutext.delete('1.0', END)
        Status="Loading... Audio from class :"
        fileslist = []

        for root, dirs, files in os.walk(path):
            for filename in files:
                if filename.endswith('.wav'):
                    Filepath = path + "/" + filename
                    fileslist.append(filename)
                    audio = MonoLoader(filename=Filepath, sampleRate=self.SampleRate)()
                    if "pain" in filename:
                        self.textStatus.delete('1.0', END)
                        self.textStatus.insert(INSERT, Status + "pain")
                        print(Filepath)
                        # add to the pool with the name of pain
                        self.AudioPool["pain"].append(audio)
                    if "deaf" in filename:
                        self.textStatus.delete('1.0', END)
                        self.textStatus.insert(INSERT, Status + "deaf")
                        # add to the pool with the name of deaf
                        self.AudioPool["deaf"].append(audio)
                    if "hunger" in filename:
                        self.textStatus.delete('1.0', END)
                        self.textStatus.insert(INSERT, Status + "hunger")
                        # add to the pool with the name of hunger
                        self.AudioPool["hunger"].append(audio)
                    if "normal" in filename:
                        self.textStatus.delete('1.0', END)
                        self.textStatus.insert(INSERT, Status + "normal")
                        # add to the pool with the name of normal
                        self.AudioPool["normal"].append(audio)
                    if "asphyxia" in filename:
                        self.textStatus.delete('1.0', END)
                        self.textStatus.insert(INSERT, Status + "asphyxia")
                        # add to the pool with the name of asphyxia
                        self.AudioPool["asphyxia"].append(audio)

        self.textStatus.delete('1.0', END)
        self.textStatus.insert(INSERT, "Loading Audio Finished !")

    # ******************************************************************************************************************
    # a-Loaded Audio files Segmentation function: with segmentation size as input parameter
    # ******************************************************************************************************************
    def Segmentation(self, canvas, ax, segLenght=10):

        # instantiation
        self.setparam()
        Status = "Loaded Audio Segmentation...:"
        # Functions initialisation
        self.DurationI = Duration(sampleRate=self.SampleRate)
        ax.clear()
        start = [] # list of start times for segementation
        stop = [] # list of stop times for segementation
        vectsegaudio = []

        #Scan each class of audio and its containing audio files
        for C, Audiolist in self.AudioPool.iteritems():
            print(C)
            # show the segmeneted class on application text status
            self.textStatus.delete('1.0', END)
            self.textStatus.insert(INSERT, Status + C)

            for A in Audiolist:  # loop of audio list scan
                # compute the stop and start vectors
                length = int(self.DurationI(A)) # get the duration of audio

                if length > segLenght: # if length is greater then input segment needed do
                    slices = range(0, length, segLenght) # create a vector of sliced segment
                    if (length - slices[-1]) == segLenght:
                        slices.append(length)
                    else:
                        slices.append(length)
                    for i, e in enumerate(slices):
                        start.append(slices[i])
                        stop.append(slices[i + 1])
                        if slices[i] == slices[-2]:
                            break
                    #Instatiate Slicer Algorithme
                    SlicerI = Slicer(sampleRate=self.SampleRate, startTimes=start, endTimes=stop)
                    vectsegaudio = SlicerI(A)
                    print(C)
                    #Save segmeneted Audio in SegAudioPool
                    for seg in vectsegaudio:
                        self.SegAudioPool[C].append(seg)

                        #Plot Segmeted audio on the graphical axis
                        ax.plot(seg)
                        canvas.draw()
                        ax.clear()
                    del start[:]
                    del stop[:]
                else:
                    self.SegAudioPool[C].append(A)
                    ax.plot(A)
                    canvas.draw()
                    ax.clear()
        self.textStatus.delete('1.0', END)
        self.textStatus.insert(INSERT, "Audio Segmentation Finished")


    # ******************************************************************************************************************
    # b-Segmented Audio files Filtring function: with Cutt frequence in graphical interface input parameter
    # ******************************************************************************************************************
    def Filtering(self,canvas, ax):

        # instantiation
        self.setparam()

        #Instantiation of algorithmes
        self.FiltrHP = HighPass(cutoffFrequency=self.FreqCut, sampleRate=self.SampleRate)
        self.FiltrLP = LowPass(cutoffFrequency=4000, sampleRate=self.SampleRate)

        #Show Status
        self.textStatus.delete('1.0', END)
        self.textStatus.insert(INSERT, "Start Audio Filtring... ")

        #Scan Segmented audio Pool
        for C, Seglist in self.SegAudioPool.iteritems():
            for seg in Seglist:  # loop of audio list scan
                FLseg = self.FiltrLP(seg)
                FHseg = 4* self.FiltrHP(FLseg)

                #Save filtred signal Audio in FsegPool
                self.FsegPool[C].append(FHseg)
                #ax.plot(FHseg)
                #canvas.draw()
                #ax.clear()

        self.textStatus.delete('1.0', END)
        self.textStatus.insert(INSERT, "Audio Filtring Finished ! ")


    # ******************************************************************************************************************
    # c-Filtred Audio Pre-Processing function: remove silence and low energy parts
    # ******************************************************************************************************************
    def preprocessing(self,canvas, ax):

        # instantiation
        self.setparam()

        #Show Status
        self.textStatus.delete('1.0', END)
        self.textStatus.insert(INSERT, "Start Audio Preprocessing... ")

        self.Getwindowing = Windowing(size=self.WinSize, type=self.winType)
        self.Getenergy = Energy()

        # Variable initialisation
        Energyt = []
        E_th = np.array([0.])

        #  calculate the energy threshold
        for C, FSegVect in self.FsegPool.iteritems():
            for Fseg in FSegVect:  # loop of audio list scan
                #  calculate the energy threshold
                for frame in FrameGenerator(Fseg, frameSize=self.WinSize, hopSize=self.hopSize, startFromZero=True):
                    Energyt.append(self.Getenergy(self.Getwindowing(frame)))
                E_th = np.mean(Energyt) / 4.0
                self.Energythreshold[C].append(E_th)

        Energyt = []

        for C, FSegVect in self.FsegPool.iteritems():
            for cntr, Fseg in enumerate(FSegVect):  # loop of audio list scan
                #  Compare and eliminate silence
                for frame in FrameGenerator(Fseg, frameSize=self.WinSize, hopSize=self.hopSize, startFromZero=True):
                    Energyt = np.array(self.Getenergy(self.Getwindowing(frame)))
                    if Energyt > self.Energythreshold[C][cntr]:
                        self.PAudioSegPool[C].append(frame)
                        #ax.plot(frame)
                        #canvas.draw()
                        #ax.clear()

        self.textStatus.delete('1.0', END)
        self.textStatus.insert(INSERT, "Audio Preprocessing... Finished ! ")



    # ******************************************************************************************************************
    # d-Features Extraction function
    # ******************************************************************************************************************
    def Extractfeature(self,canvas, ax):

        #instantiation
        self.setparam()

        # Essentia Algorithme instantiation
        self.textStatus.delete('1.0', END)
        self.textStatus.insert(INSERT, "Start Audio Features Extraction... ")
        self.Getwindowing = Windowing(size=self.WinSize, type=self.winType)
        self.Getenergy = Energy()
        self.Getspectrum = Spectrum()
        self.getCentralMoments = CentralMoments()
        self.getDistributionShape = DistributionShape()
        self.Getmfcc = MFCC()
        self.GetLPC = LPC()
        self.getLoudness = Loudness()
        self.GetSilenceRate = SilenceRate()
        self.GetEnvelope = Envelope()
        self.GetFlatnessSFX = FlatnessSFX()
        self.TFrequency = TuningFrequencyExtractor(frameSize=self.WinSize, hopSize=self.hopSize)
        self.getHFC = HFC(sampleRate=self.SampleRate)
        self.HPCP = HPCP(sampleRate=self.SampleRate, nonLinear=True)
        self.GetZCR = ZeroCrossingRate()
        self.SpeqPeaks = SpectralPeaks()

        #Scan Pre-Processed audio and extract features
        for C, PAudioSeg in self.PAudioSegPool.iteritems():
            for frame in PAudioSeg:
                mfcc_bds, mfcc_CF = self.Getmfcc(self.Getspectrum(self.Getwindowing(frame)))
                LPC_CF, LPC_rfl = self.GetLPC(self.Getwindowing(frame))
                freq, magn = self.SpeqPeaks(self.Getspectrum(self.Getwindowing(frame)))
                hpcp = self.HPCP(freq, magn)  # vector
                crlMmntC = self.getDistributionShape(self.getCentralMoments(self.Getwindowing(frame)))  # vector
                ZCRC = self.GetZCR(self.Getwindowing(frame))  # real
                LdnssC = self.getLoudness(self.Getwindowing(frame))  # real
                FltnssSFXC = self.GetFlatnessSFX(self.GetEnvelope(self.Getwindowing(frame)))  # real
                TungFreq = np.mean(self.TFrequency(self.Getwindowing(frame)))  # real big value
                # bpm,ticks,estim,bpmi = self.GetRhythmExtractor(self.Getwindowing(frame))
                hfc = self.getHFC(self.Getspectrum(self.Getwindowing(frame)))  # real big value
                Featuers = np.concatenate((mfcc_CF,mfcc_bds,LPC_rfl, LPC_CF, hpcp, crlMmntC[0:3:2], [ZCRC],[LdnssC], [FltnssSFXC]))
                #FeatuersT = essentia.array(Featuers).T
                #ax.plot(FeatuersT)
                #canvas.draw()
                #ax.clear()

                self.Feautures[C].append(Featuers)

        self.textStatus.delete('1.0', END)
        self.textStatus.insert(INSERT, "Audio Features Extraction... Finished ! ")



    # ******************************************************************************************************************
    # e-Save data as table structures extracted features in .CSV file
    # ******************************************************************************************************************
    def saveData(self):

        self.textStatus.delete('1.0', END)
        self.textStatus.insert(INSERT, "Start Data Base Creation ! ")

        for C, FeaturesV in self.PAudioSegPool.iteritems():

            Tblw = essentia.array(self.Feautures[C])
            print(self.datapath)
            fw = open(self.datapath, 'a')  # create the file with the name assotiated to the path
            for i in range(len(Tblw)):  # scan the table lines
                for j in range(len(Tblw[i])):
                    fw.write(str(Tblw[i, j]) + ';')  # write the data and add space
                fw.write(C)
                fw.write('\n')  # go back the to line once the table column end
            fw.close()

        self.textStatus.delete('1.0', END)
        self.textStatus.insert(INSERT, "Data Base has been Created ! ")

    # ******************************************************************************************************************
    # f-Classification algorithmes function : depend on the selected modele name execute the algorithme
    # ******************************************************************************************************************
    def ClassificationAlgo(self,canvas,ax):

        #Show Status
        self.textStatus.delete('1.0', END)
        self.textStatus.insert(INSERT, "Start Learning... ! ")

        # Get the selected model
        Modelname=self.algoselected
        Labels = ['pain', 'normal', 'asphyxia', 'hunger', 'deaf']

        # Load data from .CSV file and save it in matrix
        with open(self.datapath) as csvfile:
            dataset = csv.reader(csvfile, delimiter=';')
            X = []
            Y = []
            for data in dataset:
                X.append([float(x) for x in data[:len(data) - 1]])
                Y.append([data[len(data) - 1]])
            XX = np.array(X)
            YY = np.array(Y)

        # Convert string labels to numeric labels
        NumLabl = []
        for Cl in YY:
            for c in Labels:
                if c == Cl:
                    NumLabl.append(Labels.index(c))
        YYN = np.array(NumLabl).T

        # data split to training and test partitions
        Per = 70  # training percentage
        Fnbr = len(XX)  # feauters number
        PTr = int(Fnbr * Per / 100)  # training part

        # random index generation for features lines on dataset
        indrd = random.sample(range(0, Fnbr), Fnbr)
        X_train = XX[indrd[0:PTr]]
        # Y_train=YY[indrd[0:PTr]].ravel()
        Y_trainN = YYN[indrd[0:PTr]].ravel()
        X_test = XX[indrd[PTr:Fnbr]]
        # Y_test = YY[indrd[PTr:Fnbr]].ravel()
        Y_testN = YYN[indrd[PTr:Fnbr]].ravel()

        #print(X_train.shape)
        #print(Y_trainN.shape)

        # chose of model to be used
        if Modelname == "KNN":

            # ***************************************
            # KNN Model
            # ***************************************
            KNNparam={'n_neighbors': 5, 'weights': 'uniform'}
            knn = neighbors.KNeighborsClassifier()
            knn.set_params(**KNNparam)
            knn.fit(X_train, Y_trainN)
            y_pred = knn.predict(X_test)

            # compare actual response values (y_test) with predicted response values (y_pred)
            self.Occutext.delete('1.0', END)
            self.Occutext.insert(INSERT, metrics.accuracy_score(Y_testN, y_pred))
            self.Precitext.delete('1.0', END)
            self.Precitext.insert(INSERT, metrics.precision_score(Y_testN, y_pred,average=None).mean())
            self.Fmestext.delete('1.0', END)
            self.Fmestext.insert(INSERT, metrics.f1_score(Y_testN, y_pred,average=None).mean())
            self.Rapptext.delete('1.0', END)
            self.Rapptext.insert(INSERT, metrics.recall_score(Y_testN, y_pred,average=None).mean())
            self.MatricCText.delete('1.0', END)
            self.MatricCText.insert(INSERT, metrics.confusion_matrix(Y_testN, y_pred))


            # ***************************************
            # KNN Cross-Validation and Parameter Fitting
            # ***************************************

            # creating odd list of K for KNN
            myList = list(range(1, 30))

            # subsetting just the odd ones
            Kneighbors = filter(lambda x: x % 2 != 0, myList)

            # empty list that will hold cv scores
            cv_scores = []

            # perform 10-fold cross validation
            for k in Kneighbors:
                knn = neighbors.KNeighborsClassifier(n_neighbors=k, weights='uniform')
                scores = model_selection.cross_val_score(knn, X_train, Y_trainN, cv=10, scoring='accuracy')
                cv_scores.append(scores.mean())

            print("cv scors ", cv_scores)
            ax.clear()
            ax.plot(Kneighbors, cv_scores)
            canvas.draw()

        elif Modelname == "NB":

            # ***************************************
            # NB Model
            # ***************************************
            NBM = naive_bayes.GaussianNB()
            NBM.fit(X_train, Y_trainN)
            y_pred = NBM.predict(X_test)

            # compare actual response values (y_test) with predicted response values (y_pred)

            self.Occutext.delete('1.0', END)
            self.Occutext.insert(INSERT, metrics.accuracy_score(Y_testN, y_pred))
            self.Precitext.delete('1.0', END)
            self.Precitext.insert(INSERT, metrics.precision_score(Y_testN, y_pred,average=None).mean())
            self.Fmestext.delete('1.0', END)
            self.Fmestext.insert(INSERT, metrics.f1_score(Y_testN, y_pred,average=None).mean())
            self.Rapptext.delete('1.0', END)
            self.Rapptext.insert(INSERT, metrics.recall_score(Y_testN, y_pred,average=None).mean())
            self.MatricCText.delete('1.0', END)
            self.MatricCText.insert(INSERT, metrics.confusion_matrix(Y_testN, y_pred))


        elif Modelname == "MLP":

            # ***************************************
            # MLP Model
            # ***************************************
            MLPparam={'alpha': 0.0001, 'activation': 'logistic', 'max_iter': 2000, 'batch_size': "auto", \
              'hidden_layer_sizes': 20, 'solver': 'lbfgs', 'verbose': 'False', 'learning_rate': 'adaptive'}

            MLPM = MLPClassifier()
            MLPM.set_params(**MLPparam)
            MLPM.fit(X_train, Y_trainN)

            y_pred = MLPM.predict(X_test)

            # compare actual response values (y_test) with predicted response values (y_pred)
            self.Occutext.delete('1.0', END)
            self.Occutext.insert(INSERT, metrics.accuracy_score(Y_testN, y_pred))
            self.Precitext.delete('1.0', END)
            self.Precitext.insert(INSERT, metrics.precision_score(Y_testN, y_pred,average=None).mean())
            self.Fmestext.delete('1.0', END)
            self.Fmestext.insert(INSERT, metrics.f1_score(Y_testN, y_pred,average=None).mean())
            self.Rapptext.delete('1.0', END)
            self.Rapptext.insert(INSERT, metrics.recall_score(Y_testN, y_pred,average=None).mean())
            self.MatricCText.delete('1.0', END)
            self.MatricCText.insert(INSERT, metrics.confusion_matrix(Y_testN, y_pred))

        elif Modelname == "SVM":

            # ***************************************
            # SVM Model
            # ***************************************

            ## Ajuste de parametros
            SVM_params = {'multi_class':'ovr','max_iter':1000}
            # SVMM = svm.SVC()
            SVMM = svm.LinearSVC()
            SVMM.set_params(**SVM_params)
            SVMM.fit(X_train, Y_trainN)

            y_pred = SVMM.predict(X_test)

            # compare actual response values (y_test) with predicted response values (y_pred)
            self.Occutext.delete('1.0', END)
            self.Occutext.insert(INSERT, metrics.accuracy_score(Y_testN, y_pred))
            self.Precitext.delete('1.0', END)
            self.Precitext.insert(INSERT, metrics.precision_score(Y_testN, y_pred,average=None).mean())
            self.Fmestext.delete('1.0', END)
            self.Fmestext.insert(INSERT, metrics.f1_score(Y_testN, y_pred,average=None).mean())
            self.Rapptext.delete('1.0', END)
            self.Rapptext.insert(INSERT, metrics.recall_score(Y_testN, y_pred,average=None).mean())
            self.MatricCText.delete('1.0', END)
            self.MatricCText.insert(INSERT, metrics.confusion_matrix(Y_testN, y_pred))


        else:
            print("The given model does not exist")

root = Tk()
app = Application(master=root)
app.mainloop()
root.destroy()