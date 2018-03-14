# Video Object Counting
This project counts the objects of interest from a video. The user input is an ROI rectangle around one of the objects in a video frame. Using HDBSCAN we then find all other objects that have the similar local features as the object in the ROI. 

## Compile
1. git clone https://github.com/ojmakhura/dataset.git
2. git clone https://github.com/ojmakhura/vocount.git
3. mkdir build
4. cd build
5. cmake ..
6. make

## Console App
- console/vocount_cli --help

## QT 5 App
- vui/vui 

