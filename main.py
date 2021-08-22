from tkinter import *
import tkinter.messagebox
from CI_project import cnn as c
from CI_project import B_model as B
import numpy as np
from PIL import ImageTk,Image
from tkinter import filedialog
from pickle import dump
import os
import tflearn
import gc
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import  conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import cv2


def open_img() :
    global path

    canvas.delete("all")

    img_path = filedialog.askopenfilename()
    path=img_path
    img = Image.open(img_path)
    img = img.resize((512, 512), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    canvas.create_image(80, 30, anchor=NW, image=img)


    testing_data=[]

    img_data = cv2.imread(path, cv2.IMREAD_COLOR)
    img_data = cv2.resize(img_data, (150, 150))
    img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
    testing_data.append([np.array(img_data)])

    test = np.array([i[0] for i in testing_data]).reshape(-1, 150, 150, 3)
    prediction = model.predict(test)

    index_max = np.argmax(prediction[0])


    header = Label(text="You Have")
    header.place(x=700, y=15)

    if index_max == 0:
        disease = Label(text="Acne")
        disease.place(x=700, y=60)

        treatment = Label(text="Keep your hands off your face! ...\nTread carefully when it comes to home remedies. ...\nApply a warm compress. ...\nUse an acne spot treatment. ...\nWash your face regularly. ...\nTry a product with salicylic acid. ...\nGo light on your makeup. ...\nTweak your diet.")
        treatment.place(x=700, y=120)
    elif index_max == 1:
        disease = Label(text="Hairloss")
        disease.place(x=700, y=60)

        treatment = Label(text="Regularly wash your hair with mild shampoo. ...\nVitamin for hair loss. ...\nEnrich diet with protein. ...\nScalp massage with essential oils. ...\nAvoid brushing wet hair. ...\nGarlic juice, onion juice or ginger juice. ...\nKeep yourself hydrated. ...\nRub green tea into your hair.")
        treatment.place(x=700, y=120)

    elif index_max == 2:
        disease = Label(text="Nail Fungus")
        disease.place(x=700, y=60)

        treatment = Label(text="try to use different shower or wear flip flops in the shower to avoid coming in contact with it\naggressive clipping of the nails can turn into portals of entry for the fungus.\nClean your nail trimmer before using it.\nDo not tear or rip your toenails on purpose.\nKeep your feet dry. Make sure to fully dry your feet after a shower.\nTrim toenails straight across (don’t round the edges).\nWear shoes that fit correctly. They should not be too loose or tight around the toes.\nIf you have diabetes, follow all foot care recommendations from your healthcare provider.")
        treatment.place(x=700, y=120)
    elif index_max == 3:
        disease = Label(text="Normal")
        disease.place(x=700, y=60)

        treatment = Label(text="")
        treatment.place(x=700, y=120)
    elif index_max == 4:
        disease = Label(text="Skin Allergy")
        disease.place(x=700, y=60)

        treatment = Label(text="Hydrocortisone cream.\nOintments like calamine lotion.\nAntihistamines.\nCold compresses.\nOatmeal baths.\nTalk to your doctor about what's best for your specific rash")
        treatment.place(x=700, y=120)
    elif index_max == 5:
        disease = Label(text="benign")
        disease.place(x=700, y=60)
        treatment = Label(text="Treatment for BPH has long been medications and procedures\nsuch as lasers or an electric loop, which burn the prostate from the inside out\nBut, now, a relatively new convective water therapy treatment uses steam to make the prostate smaller")
        treatment.place(x=700, y=120)
    elif index_max == 6:
        disease = Label(text="malignant")
        disease.place(x=700, y=60)

        treatment = Label(text="Surgery. The goal of surgery is to remove the cancer or as much of the cancer as possible\nChemotherapy. Chemotherapy uses drugs to kill cancer cells\nRadiation therapy. ...\nBone marrow transplant. ...\nImmunotherapy. ...\nHormone therapy. ...\nTargeted drug therapy. ...\nCryoablation.")
        treatment.place(x=700, y=120)


    gc.collect()
    form.mainloop()



def evaluate(IMG_SIZE):
    global model
    conv_input = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

    conv1 = conv_2d(conv_input, 32, 5, regularizer='L1', weight_decay=0.0, activation='relu')
    pool1 = max_pool_2d(conv1, 5)

    conv2 = conv_2d(pool1, 64, 5, activation='relu')
    pool2 = max_pool_2d(conv2, 5)

    conv3 = conv_2d(pool2, 128, 5, activation='relu')
    pool3 = max_pool_2d(conv3, 5)

    conv4 = conv_2d(pool3, 64, 5, activation='relu')
    pool4 = max_pool_2d(conv4, 5)

    conv5 = conv_2d(pool4, 32, 5, activation='relu')
    pool5 = max_pool_2d(conv5, 5)

    fully_layer = fully_connected(pool5, 1024, activation='relu')
    fully_layer = dropout(fully_layer, 0.5)

    cnn_layers = fully_connected(fully_layer, 7, activation='softmax')
    cnn_layers = regression(cnn_layers, optimizer='Adam', learning_rate=.001, loss='categorical_crossentropy',
                            name='targets')
    model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3)

    model.load('model.tfl')

    return model


def main():

    model = evaluate(150)

    #TRAIN_DIR,TEST_DIR=c.start()
    #X_train,y_train,test=c.read(TRAIN_DIR,TEST_DIR,150)

    #vgg_model=c.transfer_learn(X_train,y_train,50)
    #dump(vgg_model, open('vgg_model.pkl', 'wb'))

    ################GUI##################

    Upload_butt = Button(text="   Upload   ", command=open_img)
    Upload_butt.place(x=550, y=600)


    form.mainloop()
    ######################################


    #data_gen=c.data_augmentation(X_train)
    #print(np.shape(X_train))
    #print(data_gen)
    #temp_model=B.pure_cnn_model(X_train,y_train,50)

    #model=c.built_model(X_train,y_train,150,'CI_Project_cnn',0.001)
    #c.plot(history)

    #c.test_res(test,model,150,TEST_DIR)


if __name__ == '__main__':
    form = Tk()
    canvas = Canvas(form, width=1200, height=650)
    canvas.pack()
    main()


