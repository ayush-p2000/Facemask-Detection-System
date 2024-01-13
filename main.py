import streamlit as st
import cv2
from keras.models import load_model
import numpy as np
st.set_page_config(page_title="Face Mask Detection System",page_icon="https://cdn-icons-png.flaticon.com/512/5985/5985970.png")
st.title("FACE MASK DETECTION SYSTEM")
st.sidebar.image("https://cdn.hackernoon.com/images/oO6rUouOWRYlzw88QM9pb0KyMIJ3-bxfy3m27.png")
choice=st.sidebar.selectbox("My Menu",("HOME","URL","CAMERA","FEEDBACK"))
if(choice=="HOME"):
    st.image("https://web.q-better.com/wp-content/uploads/2020/12/03-image-Mask.png")
    st.markdown("<center><h1>WELCOME<h1><center>",unsafe_allow_html=True)
    st.write("This is a Computer Vision Application which can detect whether the person is wearing a mask or not.This Application access data from Web Camera,IP Camera.It also stores the data of those who are not wearing the mask.")
    st.write("This Application can be used in Hospitals,Reasearch Labs,Polluted Areas ,Air borne pandemic.")
elif(choice=="URL"):
    url=st.text_input("Enter Video URL Here")
    btn=st.button("Start Detection")
    window=st.empty()
    if btn:
        facemodel=cv2.CascadeClassifier("face.xml")
        maskmodel=load_model("mask.h5",compile=False)
        vid=cv2.VideoCapture(url)
        i=1
        btn2=st.button("Stop Detection")
        if btn2:
            vid.release()
            st.experimental_rerun()
        while(vid.isOpened()):
            flag,frame=vid.read()            
            if(flag):
                pred=facemodel.detectMultiScale(frame)
                for (x,y,l,w) in pred:            
                    face_img=frame[y:y+w,x:x+l]
                    face_img = cv2.resize(face_img, (224, 224), interpolation=cv2.INTER_AREA) 
                    face_img = np.asarray(face_img, dtype=np.float32).reshape(1, 224, 224, 3)
                    face_img = (face_img / 127.5) - 1
                    pred = maskmodel.predict(face_img)[0][0]
                    if(pred>0.9):
                        path="Data/"+str(i)+".jpg"
                        i=i+1
                        cv2.imwrite(path,frame[y:y+w,x:x+l])
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),4)
                    else:
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),4) 
                window.image(frame,channels="BGR")
elif(choice=="CAMERA"):
    cam=st.selectbox("Select 0 for Primary Camera and 1 for Secondary Camera",("None",0,1))
    btn=st.button("Start Detection")
    window=st.empty()
    if btn:
        facemodel=cv2.CascadeClassifier("face.xml")
        maskmodel=load_model("mask.h5",compile=False)
        vid=cv2.VideoCapture(cam)
        i=1
        btn2=st.button("Stop Detection")
        if btn2:
            vid.release()
            st.experimental_rerun()
        while(vid.isOpened()):
            flag,frame=vid.read()            
            if(flag):
                pred=facemodel.detectMultiScale(frame)
                for (x,y,l,w) in pred:            
                    face_img=frame[y:y+w,x:x+l]
                    face_img = cv2.resize(face_img, (224, 224), interpolation=cv2.INTER_AREA) 
                    face_img = np.asarray(face_img, dtype=np.float32).reshape(1, 224, 224, 3)
                    face_img = (face_img / 127.5) - 1
                    pred = maskmodel.predict(face_img)[0][0]
                    if(pred>0.9):
                        path="Data/"+str(i)+".jpg"
                        i=i+1
                        cv2.imwrite(path,frame[y:y+w,x:x+l])
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),4)
                    else:
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),4) 
                window.image(frame,channels="BGR") 
elif(choice=="FEEDBACK"):
    st.markdown('<iframe src="https://docs.google.com/forms/d/e/1FAIpQLSf9WyuoWxZnk3W20kFVcyGLAf0-1AYXnz7YAqA9gxGbSFPIbQ/viewform?embedded=true" width="640" height="1085" frameborder="0" marginheight="0" marginwidth="0">Loading…</iframe>',unsafe_allow_html=True) 

