# Video Object Counting
This project counts the objects of interest from a video. The user input is an ROI rectangle around one of the objects in a video frame. Using HDBSCAN we then find all other objects that have the similar local features as the object in the ROI. 

## Compile
1. git clone https://github.com/ojmakhura/vocount.git
2. cd vocount/thirdparty
3. ./hdbscan.sh
4. cd ..
5. mkdir build
6. cd build
7. cmake ..
8. make

## Console App
- console/vocount_cli --help

## QT 5 App
- vui/vui 

