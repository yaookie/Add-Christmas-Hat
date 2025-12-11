# Add a Christmas Hat to Everyone in the Picture 给图片中的每个人添加圣诞帽

## 1. Introduction 介绍

A Python tool that uses Dlib's 68-point facial landmark detection to automatically add a Christmas hat to every person
in an image. The hat size/position adapts to face size.

基于 Dlib 68 点人脸关键点检测的 Python 工具，可自动为图片中所有人脸添加圣诞帽，帽子大小/位置根据人脸尺寸自适应调整。

## 2. Environment Dependencies 环境依赖

Install the following packages before running:

运行前需安装以下依赖包：（dlib包如果安装失败，可尝试安装预编译版本，这里直接使用的预编译版本）

```bash
pip install opencv-python  # 图像处理
pip install dlib-bin       # 人脸检测与关键点定位
