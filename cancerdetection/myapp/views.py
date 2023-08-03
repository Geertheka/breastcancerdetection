from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
import os
from django.core.files.storage import FileSystemStorage
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import pickle
import warnings
warnings.filterwarnings("ignore")
media='media'
svc=pickle.load(open('SVM_modell.sav','rb'))
dt=pickle.load(open('dt_model.sav','rb'))
lr=pickle.load(open('LR_model.sav','rb'))
cnn=tf.keras.models.load_model('cnnkmodel.h5')
def makepredictions(path):
    c_img = cv2.imread(path, cv2.IMREAD_COLOR)
    Xi= cv2.resize(c_img, (50, 50), interpolation = cv2.INTER_LINEAR)
    X=[]
    X.append(Xi)
    X=np.array(X)
    X = X.reshape(len(X), -1)
    sv= svc.predict(X)
    l=lr.predict(X)
    d=dt.predict(X)
    a=[[sv[0],d[0],l[0]]]
    a=np.array(a)
    c=cnn.predict(a)
    p = (c > 0.5).astype("int32")
    p=p[0][0]
    print(p)
    if p == 0:
        a="Result: NO CANCER DETECTED"
    else:
        a="Result: CANCER DETECTED"
    return a
def welcome(request):
    if request.method == "POST" and 'upload' in request.FILES:
        upload = request.FILES['upload']
        if not upload:
            err = 'No image selected'
            return render(request, 'index.html', {'err': err})
        fss = FileSystemStorage()
        file = fss.save(upload.name, upload)
        file_url = fss.url(file)
        predictions=makepredictions(os.path.join(settings.MEDIA_ROOT,file))
        return render(request, 'index.html', {'pred': predictions, 'file_url': file_url})
    else:
        return render(request, 'index.html')