#-------------------------------------------------
#
# Project created by QtCreator 2018-02-10T12:15:26
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = vui
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
        src/main.cpp \
        src/vuiwindow.cpp \
        src/vuiplayer.cpp \
        ../src/print_utils.cpp \
        ../src/process_frame.cpp

HEADERS += \
        include/vuiplayer.h \
        include/vuiwindow.h

FORMS += \
        forms/vuiwindow.ui

INCLUDEPATH += "include" \
            "../include" \
            "/usr/local/opencv/include" \
            "../thirdparty/include" \
            "/usr/include/glib-2.0" \
            "/usr/lib64/glib-2.0/include"

LIBS += -L/usr/local/opencv/lib64 \
        -lopencv_core \
        -lopencv_imgcodecs \
        -lopencv_highgui \
        -lopencv_xfeatures2d \
        -lopencv_features2d \
        -lopencv_imgproc \
        -lopencv_videoio

LIBS += -lglib-2.0

LIBS += -L../thirdparty/lib \
        -lhdbscan

QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS +=  -fopenmp

QMAKE_CXXFLAGS+= -std=c++11 \
                -O3 -march=native
QMAKE_LFLAGS +=  -std=c++11 \
                -O3 -march=native

RESOURCES += \
    icons.qrc
