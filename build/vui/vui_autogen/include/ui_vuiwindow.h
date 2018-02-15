/********************************************************************************
** Form generated from reading UI file 'vuiwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.9.4
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_VUIWINDOW_H
#define UI_VUIWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSplitter>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_VUIWindow
{
public:
    QAction *actionStop;
    QAction *actionExit;
    QAction *actionPlay;
    QAction *actionPause;
    QWidget *centralWidget;
    QWidget *layoutWidget;
    QHBoxLayout *horizontalLayout_6;
    QCheckBox *descriptorSpaceBox;
    QCheckBox *filteredDescriptorBox;
    QCheckBox *imageSpaceBox;
    QWidget *layoutWidget1;
    QHBoxLayout *horizontalLayout;
    QVBoxLayout *verticalLayout;
    QLabel *videoLabel;
    QLabel *outFolderLabel;
    QLabel *truthFolderLabel;
    QVBoxLayout *verticalLayout_2;
    QLineEdit *videoSelectEdit;
    QLineEdit *outFolderSelectEdit;
    QLineEdit *truthFolderSelectEdit;
    QVBoxLayout *verticalLayout_3;
    QPushButton *videoSelectButton;
    QPushButton *outputFolderButton;
    QPushButton *truthFolderButton;
    QSplitter *splitter;
    QLabel *sampleSizeLabel;
    QLineEdit *sampleSizeEdit;
    QSplitter *splitter_2;
    QLabel *trackerLabel;
    QComboBox *trackerComboBox;
    QMenuBar *menuBar;
    QMenu *menuFile;
    QStatusBar *statusBar;
    QToolBar *mainToolBar;

    void setupUi(QMainWindow *VUIWindow)
    {
        if (VUIWindow->objectName().isEmpty())
            VUIWindow->setObjectName(QStringLiteral("VUIWindow"));
        VUIWindow->resize(616, 337);
        actionStop = new QAction(VUIWindow);
        actionStop->setObjectName(QStringLiteral("actionStop"));
        actionStop->setEnabled(false);
        QIcon icon;
        icon.addFile(QStringLiteral(":/icons/icons/stop1.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionStop->setIcon(icon);
        actionExit = new QAction(VUIWindow);
        actionExit->setObjectName(QStringLiteral("actionExit"));
        QIcon icon1;
        icon1.addFile(QStringLiteral(":/icons/icons/exit1.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionExit->setIcon(icon1);
        actionPlay = new QAction(VUIWindow);
        actionPlay->setObjectName(QStringLiteral("actionPlay"));
        actionPlay->setEnabled(false);
        QIcon icon2;
        icon2.addFile(QStringLiteral(":/icons/icons/play2.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionPlay->setIcon(icon2);
        actionPause = new QAction(VUIWindow);
        actionPause->setObjectName(QStringLiteral("actionPause"));
        actionPause->setEnabled(false);
        QIcon icon3;
        icon3.addFile(QStringLiteral(":/icons/icons/pause1.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionPause->setIcon(icon3);
        centralWidget = new QWidget(VUIWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        layoutWidget = new QWidget(centralWidget);
        layoutWidget->setObjectName(QStringLiteral("layoutWidget"));
        layoutWidget->setGeometry(QRect(100, 210, 451, 25));
        horizontalLayout_6 = new QHBoxLayout(layoutWidget);
        horizontalLayout_6->setSpacing(6);
        horizontalLayout_6->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_6->setObjectName(QStringLiteral("horizontalLayout_6"));
        horizontalLayout_6->setContentsMargins(0, 0, 0, 0);
        descriptorSpaceBox = new QCheckBox(layoutWidget);
        descriptorSpaceBox->setObjectName(QStringLiteral("descriptorSpaceBox"));
        descriptorSpaceBox->setChecked(true);

        horizontalLayout_6->addWidget(descriptorSpaceBox);

        filteredDescriptorBox = new QCheckBox(layoutWidget);
        filteredDescriptorBox->setObjectName(QStringLiteral("filteredDescriptorBox"));

        horizontalLayout_6->addWidget(filteredDescriptorBox);

        imageSpaceBox = new QCheckBox(layoutWidget);
        imageSpaceBox->setObjectName(QStringLiteral("imageSpaceBox"));

        horizontalLayout_6->addWidget(imageSpaceBox);

        layoutWidget1 = new QWidget(centralWidget);
        layoutWidget1->setObjectName(QStringLiteral("layoutWidget1"));
        layoutWidget1->setGeometry(QRect(10, 20, 551, 91));
        horizontalLayout = new QHBoxLayout(layoutWidget1);
        horizontalLayout->setSpacing(6);
        horizontalLayout->setContentsMargins(11, 11, 11, 11);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        verticalLayout = new QVBoxLayout();
        verticalLayout->setSpacing(6);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        videoLabel = new QLabel(layoutWidget1);
        videoLabel->setObjectName(QStringLiteral("videoLabel"));

        verticalLayout->addWidget(videoLabel);

        outFolderLabel = new QLabel(layoutWidget1);
        outFolderLabel->setObjectName(QStringLiteral("outFolderLabel"));

        verticalLayout->addWidget(outFolderLabel);

        truthFolderLabel = new QLabel(layoutWidget1);
        truthFolderLabel->setObjectName(QStringLiteral("truthFolderLabel"));

        verticalLayout->addWidget(truthFolderLabel);


        horizontalLayout->addLayout(verticalLayout);

        verticalLayout_2 = new QVBoxLayout();
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        videoSelectEdit = new QLineEdit(layoutWidget1);
        videoSelectEdit->setObjectName(QStringLiteral("videoSelectEdit"));

        verticalLayout_2->addWidget(videoSelectEdit);

        outFolderSelectEdit = new QLineEdit(layoutWidget1);
        outFolderSelectEdit->setObjectName(QStringLiteral("outFolderSelectEdit"));

        verticalLayout_2->addWidget(outFolderSelectEdit);

        truthFolderSelectEdit = new QLineEdit(layoutWidget1);
        truthFolderSelectEdit->setObjectName(QStringLiteral("truthFolderSelectEdit"));

        verticalLayout_2->addWidget(truthFolderSelectEdit);


        horizontalLayout->addLayout(verticalLayout_2);

        verticalLayout_3 = new QVBoxLayout();
        verticalLayout_3->setSpacing(6);
        verticalLayout_3->setObjectName(QStringLiteral("verticalLayout_3"));
        videoSelectButton = new QPushButton(layoutWidget1);
        videoSelectButton->setObjectName(QStringLiteral("videoSelectButton"));
        QIcon icon4;
        icon4.addFile(QStringLiteral(":/icons/icons/video2.png"), QSize(), QIcon::Normal, QIcon::Off);
        videoSelectButton->setIcon(icon4);

        verticalLayout_3->addWidget(videoSelectButton);

        outputFolderButton = new QPushButton(layoutWidget1);
        outputFolderButton->setObjectName(QStringLiteral("outputFolderButton"));
        QIcon icon5;
        icon5.addFile(QStringLiteral(":/icons/icons/open_folder.png"), QSize(), QIcon::Normal, QIcon::Off);
        outputFolderButton->setIcon(icon5);

        verticalLayout_3->addWidget(outputFolderButton);

        truthFolderButton = new QPushButton(layoutWidget1);
        truthFolderButton->setObjectName(QStringLiteral("truthFolderButton"));
        truthFolderButton->setIcon(icon5);

        verticalLayout_3->addWidget(truthFolderButton);


        horizontalLayout->addLayout(verticalLayout_3);

        splitter = new QSplitter(centralWidget);
        splitter->setObjectName(QStringLiteral("splitter"));
        splitter->setGeometry(QRect(110, 130, 121, 25));
        splitter->setOrientation(Qt::Horizontal);
        sampleSizeLabel = new QLabel(splitter);
        sampleSizeLabel->setObjectName(QStringLiteral("sampleSizeLabel"));
        splitter->addWidget(sampleSizeLabel);
        sampleSizeEdit = new QLineEdit(splitter);
        sampleSizeEdit->setObjectName(QStringLiteral("sampleSizeEdit"));
        splitter->addWidget(sampleSizeEdit);
        splitter_2 = new QSplitter(centralWidget);
        splitter_2->setObjectName(QStringLiteral("splitter_2"));
        splitter_2->setGeometry(QRect(100, 170, 221, 25));
        splitter_2->setOrientation(Qt::Horizontal);
        trackerLabel = new QLabel(splitter_2);
        trackerLabel->setObjectName(QStringLiteral("trackerLabel"));
        splitter_2->addWidget(trackerLabel);
        trackerComboBox = new QComboBox(splitter_2);
        trackerComboBox->setObjectName(QStringLiteral("trackerComboBox"));
        trackerComboBox->setMaxVisibleItems(24);
        splitter_2->addWidget(trackerComboBox);
        VUIWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(VUIWindow);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 616, 22));
        menuFile = new QMenu(menuBar);
        menuFile->setObjectName(QStringLiteral("menuFile"));
        VUIWindow->setMenuBar(menuBar);
        statusBar = new QStatusBar(VUIWindow);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        VUIWindow->setStatusBar(statusBar);
        mainToolBar = new QToolBar(VUIWindow);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        mainToolBar->setEnabled(true);
        VUIWindow->addToolBar(Qt::TopToolBarArea, mainToolBar);

        menuBar->addAction(menuFile->menuAction());
        menuFile->addAction(actionPlay);
        menuFile->addAction(actionPause);
        menuFile->addAction(actionStop);
        menuFile->addSeparator();
        menuFile->addAction(actionExit);
        mainToolBar->addAction(actionPlay);
        mainToolBar->addAction(actionPause);
        mainToolBar->addAction(actionStop);
        mainToolBar->addSeparator();
        mainToolBar->addAction(actionExit);

        retranslateUi(VUIWindow);

        QMetaObject::connectSlotsByName(VUIWindow);
    } // setupUi

    void retranslateUi(QMainWindow *VUIWindow)
    {
        VUIWindow->setWindowTitle(QApplication::translate("VUIWindow", "MainWindow", Q_NULLPTR));
        actionStop->setText(QApplication::translate("VUIWindow", "&Stop", Q_NULLPTR));
#ifndef QT_NO_SHORTCUT
        actionStop->setShortcut(QApplication::translate("VUIWindow", "Ctrl+S", Q_NULLPTR));
#endif // QT_NO_SHORTCUT
        actionExit->setText(QApplication::translate("VUIWindow", "&Exit", Q_NULLPTR));
#ifndef QT_NO_SHORTCUT
        actionExit->setShortcut(QApplication::translate("VUIWindow", "Ctrl+X", Q_NULLPTR));
#endif // QT_NO_SHORTCUT
        actionPlay->setText(QApplication::translate("VUIWindow", "&Play", Q_NULLPTR));
        actionPlay->setIconText(QApplication::translate("VUIWindow", "&Play", Q_NULLPTR));
#ifndef QT_NO_TOOLTIP
        actionPlay->setToolTip(QApplication::translate("VUIWindow", "Play", Q_NULLPTR));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_SHORTCUT
        actionPlay->setShortcut(QApplication::translate("VUIWindow", "Ctrl+P", Q_NULLPTR));
#endif // QT_NO_SHORTCUT
        actionPause->setText(QApplication::translate("VUIWindow", "Pa&use", Q_NULLPTR));
#ifndef QT_NO_TOOLTIP
        actionPause->setToolTip(QApplication::translate("VUIWindow", "Pause Video", Q_NULLPTR));
#endif // QT_NO_TOOLTIP
        descriptorSpaceBox->setText(QApplication::translate("VUIWindow", "Descriptor Space", Q_NULLPTR));
        filteredDescriptorBox->setText(QApplication::translate("VUIWindow", "Filtered Descriptor", Q_NULLPTR));
        imageSpaceBox->setText(QApplication::translate("VUIWindow", "Image Space", Q_NULLPTR));
        videoLabel->setText(QApplication::translate("VUIWindow", "Video", Q_NULLPTR));
        outFolderLabel->setText(QApplication::translate("VUIWindow", "Output Folder", Q_NULLPTR));
        truthFolderLabel->setText(QApplication::translate("VUIWindow", "Truth Folder", Q_NULLPTR));
#ifndef QT_NO_TOOLTIP
        videoSelectButton->setToolTip(QApplication::translate("VUIWindow", "Select Video File", Q_NULLPTR));
#endif // QT_NO_TOOLTIP
        videoSelectButton->setText(QString());
#ifndef QT_NO_TOOLTIP
        outputFolderButton->setToolTip(QApplication::translate("VUIWindow", "Select Output Folder", Q_NULLPTR));
#endif // QT_NO_TOOLTIP
        outputFolderButton->setText(QString());
#ifndef QT_NO_TOOLTIP
        truthFolderButton->setToolTip(QApplication::translate("VUIWindow", "Select Truth Folder", Q_NULLPTR));
#endif // QT_NO_TOOLTIP
        truthFolderButton->setText(QString());
        sampleSizeLabel->setText(QApplication::translate("VUIWindow", "Sample Size", Q_NULLPTR));
        sampleSizeEdit->setText(QApplication::translate("VUIWindow", "1", Q_NULLPTR));
        trackerLabel->setText(QApplication::translate("VUIWindow", "Tracker", Q_NULLPTR));
        trackerComboBox->setCurrentText(QString());
        menuFile->setTitle(QApplication::translate("VUIWindow", "Fi&le", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class VUIWindow: public Ui_VUIWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_VUIWINDOW_H
