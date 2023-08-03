from django.http import HttpResponse
from django.shortcuts import render

def webpage1(request):
    return render(request,'index.html')
def webpage2(request):
    return render(request,'page2.html')  
def webpage3(request):
    return render(request,'page3.html')