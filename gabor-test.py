#!/usr/bin/env python
# -*- coding:utf-8 -*-
import cv2
import numpy as np 
import pylab as pl 
# gabor滤波器的构建，使用6个尺度分四个方向
# 构建Gabor滤波器
def build_filters():
	filters=[]
	#gabor尺度 6个
	ksize=[7,9,11,13,15,17]
	lamda=np.pi/2.0

	for theta in np.arange(0,np.pi,np.pi/4):
		for k in xrange(6):
			kern=cv2.getGaborKernel((ksize[k],ksize[k]),1.0,theta,lamda,0.5,0,ktype=cv2.CV_32F)
			kern/=1.5*kern.sum()
			filters.append(kern)
	return filters
#显示滤波器核示意图
def build_filters_r():
	res=[]
	#gabor尺度 6个
	ksize=[7,9,11,13,15,17]
	lamda=np.pi/2.0
	"""角度(angle) 有两种表示，度(degree) 和 弧度(radian). 
	弧度:度 = pi:180 = 3.1415:180
	这里计算的是弧度.
	所以想计算正弦30°需要np.sin(30*np.pi/180)"""
	for theta in np.arange(0,np.pi,np.pi/4): #gabor方向，0°，45°，90°，135°，共四个
		for k in xrange(6):
			#cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
			kern=cv2.getGaborKernel((ksize[k],ksize[k]),1.0,theta,lamda,0.5,0,ktype=cv2.CV_32F)
			kern/=1.5*kern.sum()
			res.append(np.asarray(kern))
	for temp in xrange(len(res)):
		pl.subplot(4,6,temp+1)
		pl.imshow(res[temp],cmap='gray')
	pl.show()
# Gabor 滤波过程
def process(img,filters):
	accum=np.zeros_like(img)
	for kern in filters:
		fimg=cv2.filter2D(img,cv2.CV_8UC3,kern)
		np.maximum(accum,fimg,accum)
	return accum

def getGabor(img,filters):
	#滤波结果
	res=[]
	srcImage=cv2.imread(img)
	for i in xrange(len(filters)):
		res1=process(srcImage,filters[i])
		res.append(np.asarray(res1))
	pl.figure(2)
	for temp in xrange(len(res)):
		pl.subplot(4,6,temp+1)
		pl.imshow(res[temp],cmap='gray')
	pl.show()

	return res

if __name__=='__main__':
	build_filters_r()
	filters=build_filters()
	getGabor('timg.jpg',filters)
	