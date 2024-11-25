# !/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import codecs
import distutils.spawn
import os.path
import platform
import re
import sys
import subprocess
import cv2

from functools import partial
from collections import defaultdict

try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    # needed for py3+qt4
    # Ref:
    # http://pyqt.sourceforge.net/Docs/PyQt4/incompatible_apis.html
    # http://stackoverflow.com/questions/21217399/pyqt4-qtcore-qvariant-object-instead-of-a-string
    if sys.version_info.major >= 3:
        import sip

        sip.setapi('QVariant', 2)
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

from libs.combobox import ComboBox
from libs.resources import *
from libs.constants import *
from libs.utils import *
from libs.settings import Settings
from libs.shape import Shape, DEFAULT_LINE_COLOR, DEFAULT_FILL_COLOR
from libs.stringBundle import StringBundle
from libs.canvas import Canvas
from libs.zoomWidget import ZoomWidget
from libs.labelDialog import LabelDialog
from libs.colorDialog import ColorDialog
from libs.labelFile import LabelFile, LabelFileError, LabelFileFormat
from libs.toolBar import ToolBar
from libs.pascal_voc_io import PascalVocReader
from libs.pascal_voc_io import XML_EXT
from libs.yolo_io import YoloReader
from libs.yolo_io import TXT_EXT
from libs.create_ml_io import CreateMLReader
from libs.create_ml_io import JSON_EXT
from libs.ustr import ustr
from libs.hashableQListWidgetItem import HashableQListWidgetItem


###################### NEW STUFF ###############################
from libs.XLFormat_io import XLReader

from database import Database
from pathlib import Path
import yaml
import numpy as np
from raw_image_to_QImage import read_matrix, numpyQImage, scale_image_base
import XenoWareFormat as xw
###################### NEW STUFF ###############################
__appname__ = 'XLabelTool'


class WindowMixin(object):

    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            addActions(menu, actions)
        return menu

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName(u'%sToolBar' % title)
        # toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        if actions:
            addActions(toolbar, actions)
        self.addToolBar(Qt.LeftToolBarArea, toolbar)
        return toolbar


class MainWindow(QMainWindow, WindowMixin):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = list(range(3))

    def __init__(self, tags, defaultFilename=None, defaultPrefdefClassFile=None, defaultSaveDir=None,
                 task_type='Object_Detection'):
        super(MainWindow, self).__init__()
        # self.labelFormat = labelFormat
        self.setWindowTitle(__appname__)

        # Load setting in the main thread
        self.settings = Settings()
        self.settings.load()
        settings = self.settings

        # Load string bundle for i18n
        self.stringBundle = StringBundle.getBundle()
        getStr = lambda strId: self.stringBundle.getString(strId)

        # Save as Pascal voc xml
        self.defaultSaveDir = defaultSaveDir
        print(f'first self.defaultsavedir: {self.defaultSaveDir}')
        self.labelFileFormat = settings.get(SETTING_LABEL_FILE_FORMAT, LabelFileFormat.PASCAL_VOC)

        # For loading all image under a directory
        self.mImgList = []
        self.dirname = None
        self.labelHist = []
        self.lastOpenDir = None

        ################################## New Stuff ###########################################
        # self.format = labelFormat
        self.task_type = task_type
        ################################## New Stuff ###########################################

        # Whether we need to save or not.
        self.dirty = False

        self._noSelectionSlot = False
        self._beginner = True
        self.screencastViewer = self.getAvailableScreencastViewer()
        self.screencast = "https://youtu.be/p0nR2YsCY_U"

        # Load predefined classes to the list
        self.loadPredefinedClasses(defaultPrefdefClassFile)

        # Main widgets and related state.
        self.labelDialog = LabelDialog(parent=self, listItem=self.labelHist)

        self.itemsToShapes = {}
        self.shapesToItems = {}
        self.prevLabelText = ''

        listLayout = QVBoxLayout()
        listLayout.setContentsMargins(0, 0, 0, 0)

        # Create a widget for using default label
        self.useDefaultLabelCheckbox = QCheckBox(getStr('useDefaultLabel'))
        self.useDefaultLabelCheckbox.setChecked(False)
        self.defaultLabelTextLine = QLineEdit()
        useDefaultLabelQHBoxLayout = QHBoxLayout()
        useDefaultLabelQHBoxLayout.addWidget(self.useDefaultLabelCheckbox)
        useDefaultLabelQHBoxLayout.addWidget(self.defaultLabelTextLine)
        useDefaultLabelContainer = QWidget()
        useDefaultLabelContainer.setLayout(useDefaultLabelQHBoxLayout)

        # Create a widget for edit and diffc button
        self.diffcButton = QCheckBox(getStr('useDifficult'))
        self.diffcButton.setChecked(False)
        self.diffcButton.stateChanged.connect(self.btnstate)
        self.editButton = QToolButton()
        self.editButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        # Add some of widgets to listLayout
        listLayout.addWidget(self.editButton)
        listLayout.addWidget(self.diffcButton)
        listLayout.addWidget(useDefaultLabelContainer)

        
        #for distance computation
        self.shape = Shape()
        self.distances = []
        
        # for RGB Visual Transformations
        
        self.do_transforms= False
        
        
        # test initializations  for directories
        
        self.AnnotationRootDir = "/media/adarsh/USB DRIVE/Ground truth generation/test/Images/Annotation_OD_refined"
        
        # Create and add combobox for showing unique labels in group
        self.comboBox = ComboBox(self)
        listLayout.addWidget(self.comboBox)

        # Create and add a widget for showing current label items
        self.labelList = QListWidget()
        labelListContainer = QWidget()
        labelListContainer.setLayout(listLayout)
        self.labelList.itemActivated.connect(self.labelSelectionChanged)
        self.labelList.itemSelectionChanged.connect(self.labelSelectionChanged)
        self.labelList.itemDoubleClicked.connect(self.editLabel)
        # Connect to itemChanged to detect checkbox changes.
        self.labelList.itemChanged.connect(self.labelItemChanged)
        listLayout.addWidget(self.labelList)

        self.dock = QDockWidget(getStr('boxLabelText'), self)
        self.dock.setObjectName(getStr('labels'))
        self.dock.setWidget(labelListContainer)

        self.fileListWidget = QListWidget()
        self.fileListWidget.itemDoubleClicked.connect(self.fileitemDoubleClicked)
        filelistLayout = QVBoxLayout()
        filelistLayout.setContentsMargins(0, 0, 0, 0)
        filelistLayout.addWidget(self.fileListWidget)
        fileListContainer = QWidget()
        fileListContainer.setLayout(filelistLayout)
        self.filedock = QDockWidget(getStr('fileList'), self)
        self.filedock.setObjectName(getStr('files'))
        self.filedock.setWidget(fileListContainer)

        self.zoomWidget = ZoomWidget()
        self.colorDialog = ColorDialog(parent=self)

        self.canvas = Canvas(parent=self)
        self.canvas.zoomRequest.connect(self.zoomRequest)
        self.canvas.setDrawingShapeToSquare(settings.get(SETTING_DRAW_SQUARE, False))

        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scroll.verticalScrollBar(),
            Qt.Horizontal: scroll.horizontalScrollBar()
        }
        self.scrollArea = scroll
        self.canvas.scrollRequest.connect(self.scrollRequest)

        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.setDirty)
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)

        self.setCentralWidget(scroll)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.filedock)
        self.filedock.setFeatures(QDockWidget.DockWidgetFloatable)

        self.dockFeatures = QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetFloatable
        self.dock.setFeatures(self.dock.features() ^ self.dockFeatures)

        # Actions
        action = partial(newAction, self)
        quit = action(getStr('quit'), self.close,
                      'Ctrl+Q', 'quit', getStr('quitApp'))

        open = action(getStr('openFile'), self.openFile,
                      'Ctrl+O', 'open', getStr('openFileDetail'))

        opendir = action(getStr('openDir'), self.openDirDialog,
                         'Ctrl+u', 'open', getStr('openDir'))

        copyPrevBounding = action(getStr('copyPrevBounding'), self.copyPreviousBoundingBoxes,
                                  'Ctrl+v', 'paste', getStr('copyPrevBounding'))

        changeSavedir = action(getStr('changeSaveDir'), self.changeSavedirDialog,
                               'Ctrl+r', 'open', getStr('changeSavedAnnotationDir'))

        openAnnotation = action(getStr('openAnnotation'), self.openAnnotationDialog,
                                'Ctrl+Shift+O', 'open', getStr('openAnnotationDetail'))

        openNextImg = action(getStr('nextImg'), self.openNextImg,
                             'd', 'next', getStr('nextImgDetail'))

        openPrevImg = action(getStr('prevImg'), self.openPrevImg,
                             'a', 'prev', getStr('prevImgDetail'))

        verify = action(getStr('verifyImg'), self.verifyImg,
                        'space', 'verify', getStr('verifyImgDetail'))

        save = action(getStr('save'), self.saveFile,
                      'Ctrl+S', 'save', getStr('saveDetail'), enabled=False)

        ######################################## NEW STUFF #######################################
        # self.menus.file.addAction(self.saveAct)
        # self.saveAct = QAction("&Save Database", self, shortcut="Shift+S", triggered=self.save_database)

        saveDatabase = action(str('&Save Database'), self.save_database, 'Shift+S')

        ######################################## NEW STUFF #######################################

        def getFormatMeta(format):
            """
            returns a tuple containing (title, icon_name) of the selected format
            """
            if format == LabelFileFormat.PASCAL_VOC:
                return ('&PascalVOC', 'format_voc')
            elif format == LabelFileFormat.XL:
                return ('&XL', 'format_xl')
            elif format == LabelFileFormat.CREATE_ML:
                return ('&CreateML', 'format_createml')

            ############################## NEW STUFF ################################################
            # elif format == LabelFileFormat.XL:
            #     return ('&XL', 'format_xl')
            ############################## NEW STUFF ################################################

        save_format = action(getFormatMeta(self.labelFileFormat)[0],
                             self.change_format, 'Ctrl+',
                             getFormatMeta(self.labelFileFormat)[1],
                             getStr('changeSaveFormat'), enabled=True)

        saveAs = action(getStr('saveAs'), self.saveFileAs,
                        'Ctrl+Shift+S', 'save-as', getStr('saveAsDetail'), enabled=False)

        close = action(getStr('closeCur'), self.closeFile, 'Ctrl+W', 'close', getStr('closeCurDetail'))

        deleteImg = action(getStr('deleteImg'), self.deleteImg, 'Ctrl+Shift+D', 'close', getStr('deleteImgDetail'))

        resetAll = action(getStr('resetAll'), self.resetAll, None, 'resetall', getStr('resetAllDetail'))

        color1 = action(getStr('boxLineColor'), self.chooseColor1,
                        'Ctrl+L', 'color_line', getStr('boxLineColorDetail'))

        createMode = action(getStr('crtBox'), self.setCreateMode,
                            'w', 'new', getStr('crtBoxDetail'), enabled=False)
        editMode = action('&Edit\nRectBox', self.setEditMode,
                          'Ctrl+J', 'edit', u'Move and edit Boxs', enabled=False)

        create = action(getStr('crtBox'), self.createShape,
                        'w', 'new', getStr('crtBoxDetail'), enabled=False)
        delete = action(getStr('delBox'), self.deleteSelectedShape,
                        'Delete', 'delete', getStr('delBoxDetail'), enabled=False)
        copy = action(getStr('dupBox'), self.copySelectedShape,
                      'Ctrl+D', 'copy', getStr('dupBoxDetail'),
                      enabled=False)

        advancedMode = action(getStr('advancedMode'), self.toggleAdvancedMode,
                              'Ctrl+Shift+A', 'expert', getStr('advancedModeDetail'),
                              checkable=True)

        hideAll = action('&Hide\nRectBox', partial(self.togglePolygons, False),
                         'Ctrl+H', 'hide', getStr('hideAllBoxDetail'),
                         enabled=False)
        showAll = action('&Show\nRectBox', partial(self.togglePolygons, True),
                         'Ctrl+A', 'hide', getStr('showAllBoxDetail'),
                         enabled=False)

        help = action(getStr('tutorial'), self.showTutorialDialog, None, 'help', getStr('tutorialDetail'))
        showInfo = action(getStr('info'), self.showInfoDialog, None, 'help', getStr('info'))

        zoom = QWidgetAction(self)
        zoom.setDefaultWidget(self.zoomWidget)
        self.zoomWidget.setWhatsThis(
            u"Zoom in or out of the image. Also accessible with"
            " %s and %s from the canvas." % (fmtShortcut("Ctrl+[-+]"),
                                             fmtShortcut("Ctrl+Wheel")))
        self.zoomWidget.setEnabled(False)

        zoomIn = action(getStr('zoomin'), partial(self.addZoom, 10),
                        'Ctrl++', 'zoom-in', getStr('zoominDetail'), enabled=False)
        zoomOut = action(getStr('zoomout'), partial(self.addZoom, -10),
                         'Ctrl+-', 'zoom-out', getStr('zoomoutDetail'), enabled=False)
        zoomOrg = action(getStr('originalsize'), partial(self.setZoom, 100),
                         'Ctrl+=', 'zoom', getStr('originalsizeDetail'), enabled=False)
        fitWindow = action(getStr('fitWin'), self.setFitWindow,
                           'Ctrl+F', 'fit-window', getStr('fitWinDetail'),
                           checkable=True, enabled=False)
        fitWidth = action(getStr('fitWidth'), self.setFitWidth,
                          'Ctrl+Shift+F', 'fit-width', getStr('fitWidthDetail'),
                          checkable=True, enabled=False)
        # Group zoom controls into a list for easier toggling.
        zoomActions = (self.zoomWidget, zoomIn, zoomOut,
                       zoomOrg, fitWindow, fitWidth)
        self.zoomMode = self.MANUAL_ZOOM
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        edit = action(getStr('editLabel'), self.editLabel,
                      'Ctrl+E', 'edit', getStr('editLabelDetail'),
                      enabled=False)
        self.editButton.setDefaultAction(edit)

        shapeLineColor = action(getStr('shapeLineColor'), self.chshapeLineColor,
                                icon='color_line', tip=getStr('shapeLineColorDetail'),
                                enabled=False)
        shapeFillColor = action(getStr('shapeFillColor'), self.chshapeFillColor,
                                icon='color', tip=getStr('shapeFillColorDetail'),
                                enabled=False)

        labels = self.dock.toggleViewAction()
        labels.setText(getStr('showHide'))
        labels.setShortcut('Ctrl+Shift+L')

        # Label list context menu.
        labelMenu = QMenu()
        addActions(labelMenu, (edit, delete))
        self.labelList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.labelList.customContextMenuRequested.connect(
            self.popLabelListMenu)

        # Draw squares/rectangles
        self.drawSquaresOption = QAction('Draw Squares', self)
        self.drawSquaresOption.setShortcut('Ctrl+Shift+R')
        self.drawSquaresOption.setCheckable(True)
        self.drawSquaresOption.setChecked(settings.get(SETTING_DRAW_SQUARE, False))
        self.drawSquaresOption.triggered.connect(self.toogleDrawSquare)

        # Store actions for further handling.
        self.actions = struct(save=save, save_format=save_format, saveAs=saveAs, open=open, close=close,
                              resetAll=resetAll, deleteImg=deleteImg,
                              lineColor=color1, create=create, delete=delete, edit=edit, copy=copy,
                              createMode=createMode, editMode=editMode, advancedMode=advancedMode,
                              shapeLineColor=shapeLineColor, shapeFillColor=shapeFillColor,
                              zoom=zoom, zoomIn=zoomIn, zoomOut=zoomOut, zoomOrg=zoomOrg,
                              fitWindow=fitWindow, fitWidth=fitWidth,
                              zoomActions=zoomActions,
                              fileMenuActions=(
                                  open, opendir, save, saveAs, close, resetAll, quit),
                              beginner=(), advanced=(),
                              editMenu=(edit, copy, delete,
                                        None, color1, self.drawSquaresOption),
                              beginnerContext=(create, edit, copy, delete),
                              advancedContext=(createMode, editMode, edit, copy,
                                               delete, shapeLineColor, shapeFillColor),
                              onLoadActive=(
                                  close, create, createMode, editMode),
                              onShapesPresent=(saveAs, hideAll, showAll),
                              save_database=saveDatabase)  ############################## NEW STUFF ########################

        self.menus = struct(
            file=self.menu('&File'),
            edit=self.menu('&Edit'),
            view=self.menu('&View'),
            help=self.menu('&Help'),
            recentFiles=QMenu('Open &Recent'),
            labelList=labelMenu)

        # Auto saving : Enable auto saving if pressing next
        self.autoSaving = QAction(getStr('autoSaveMode'), self)
        self.autoSaving.setCheckable(True)
        self.autoSaving.setChecked(settings.get(SETTING_AUTO_SAVE, False))
        # Sync single class mode from PR#106
        self.singleClassMode = QAction(getStr('singleClsMode'), self)
        self.singleClassMode.setShortcut("Ctrl+Shift+S")
        self.singleClassMode.setCheckable(True)
        self.singleClassMode.setChecked(settings.get(SETTING_SINGLE_CLASS, False))
        self.lastLabel = None
        # Add option to enable/disable labels being displayed at the top of bounding boxes
        self.displayLabelOption = QAction(getStr('displayLabel'), self)
        self.displayLabelOption.setShortcut("Ctrl+Shift+P")
        self.displayLabelOption.setCheckable(True)
        self.displayLabelOption.setChecked(settings.get(SETTING_PAINT_LABEL, True))
        self.displayLabelOption.triggered.connect(self.togglePaintLabelsOption)

        # addActions(self.menus.file,
        #            (open, opendir, copyPrevBounding, changeSavedir, openAnnotation, self.menus.recentFiles, save, save_format, saveAs, close, resetAll, deleteImg, quit))

        ############################################ NEW STUFF #####################################
        addActions(self.menus.file,
                   (open, opendir, copyPrevBounding, changeSavedir, openAnnotation, self.menus.recentFiles, save,
                    save_format, saveAs, close, resetAll, deleteImg, quit, saveDatabase))

        ############################################ NEW STUFF #####################################

        addActions(self.menus.help, (help, showInfo))
        addActions(self.menus.view, (
            self.autoSaving,
            self.singleClassMode,
            self.displayLabelOption,
            labels, advancedMode, None,
            hideAll, showAll, None,
            zoomIn, zoomOut, zoomOrg, None,
            fitWindow, fitWidth))

        self.menus.file.aboutToShow.connect(self.updateFileMenu)

        # Custom context menu for the canvas widget:
        addActions(self.canvas.menus[0], self.actions.beginnerContext)
        addActions(self.canvas.menus[1], (
            action('&Copy here', self.copyShape),
            action('&Move here', self.moveShape)))

        self.tools = self.toolbar('Tools')
        self.actions.beginner = (
            open, opendir, changeSavedir, openNextImg, openPrevImg, verify, save, save_format, None, create, copy,
            delete, None,
            zoomIn, zoom, zoomOut, fitWindow,
            fitWidth)  ############################### saveDatabase action should be added here?

        self.actions.advanced = (
            open, opendir, changeSavedir, openNextImg, openPrevImg, save, save_format, None,
            createMode, editMode, None,
            hideAll, showAll)

        self.statusBar().showMessage('%s started.' % __appname__)
        self.statusBar().show()

        # Application state.
        self.image = QImage()
        self.filePath = ustr(defaultFilename)
        self.lastOpenDir = None
        self.recentFiles = []
        self.maxRecent = 7
        self.lineColor = None
        self.fillColor = None
        self.zoom_level = 100
        self.fit_window = False
        # Add Chris
        self.difficult = False

        ## Fix the compatible issue for qt4 and qt5. Convert the QStringList to python list
        if settings.get(SETTING_RECENT_FILES):
            if have_qstring():
                recentFileQStringList = settings.get(SETTING_RECENT_FILES)
                self.recentFiles = [ustr(i) for i in recentFileQStringList]
            else:
                self.recentFiles = recentFileQStringList = settings.get(SETTING_RECENT_FILES)

        size = settings.get(SETTING_WIN_SIZE, QSize(600, 500))
        position = QPoint(0, 0)
        saved_position = settings.get(SETTING_WIN_POSE, position)
        # Fix the multiple monitors issue
        for i in range(QApplication.desktop().screenCount()):
            if QApplication.desktop().availableGeometry(i).contains(saved_position):
                position = saved_position
                break
        self.resize(size)
        self.move(position)
        saveDir = ustr(settings.get(SETTING_SAVE_DIR, None))
        print(f'saveDir: {saveDir}')
        self.lastOpenDir = ustr(settings.get(SETTING_LAST_OPEN_DIR, None))
        if self.defaultSaveDir is None and saveDir is not None and os.path.exists(saveDir):
            self.defaultSaveDir = saveDir
            self.statusBar().showMessage('%s started. Annotation will be saved to %s' %
                                         (__appname__, self.defaultSaveDir))
            self.statusBar().show()

        self.restoreState(settings.get(SETTING_WIN_STATE, QByteArray()))
        Shape.line_color = self.lineColor = QColor(settings.get(SETTING_LINE_COLOR, DEFAULT_LINE_COLOR))
        Shape.fill_color = self.fillColor = QColor(settings.get(SETTING_FILL_COLOR, DEFAULT_FILL_COLOR))
        self.canvas.setDrawingColor(self.lineColor)
        # Add chris
        Shape.difficult = self.difficult

        def xbool(x):
            if isinstance(x, QVariant):
                return x.toBool()
            return bool(x)

        if xbool(settings.get(SETTING_ADVANCE_MODE, False)):
            self.actions.advancedMode.setChecked(True)
            self.toggleAdvancedMode()

        # Populate the File menu dynamically.
        self.updateFileMenu()

        # Since loading the file may take some time, make sure it runs in the background.
        if self.filePath and os.path.isdir(self.filePath):
            self.queueEvent(partial(self.importDirImages, self.filePath or ""))
        elif self.filePath:
            self.queueEvent(partial(self.loadFile, self.filePath or ""))

        # Callbacks:
        self.zoomWidget.valueChanged.connect(self.paintCanvas)

        self.populateModeActions()

        # Display cursor coordinates at the right of status bar
        self.labelCoordinates = QLabel('')
        self.statusBar().addPermanentWidget(self.labelCoordinates)

        # Open Dir if deafult file
        if self.filePath and os.path.isdir(self.filePath):
            self.openDirDialog(dirpath=self.filePath, silent=True)

        ################################## NEW STUFF #############################
        self.database = None
        self.min_dock_width = 80
        self.progress_bar()
        self.progress_bar_RGB()
        self.create_selection_dock()
        self.create_tags_dock({idx: tag for idx, tag in enumerate(tags, start=1)})
        # self.test_dock()
        self.test_stuff_here()
        
        ## add RGB image here
        #self.RGB_toggle = self.test_RGB().isChecked()
        
        
        self.shapes_when_file_loaded = []
        self.shapeCountBefore = 0
        ################################## NEW STUFF #############################

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Control:
            self.canvas.setDrawingShapeToSquare(False)

        ###################### NEW STUFF #############################
        elif event.key() == QtCore.Qt.Key_Return:
            self.test_stuff_callback()
        ###################### NEW STUFF #############################

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Control:
            # Draw rectangle if Ctrl is pressed
            self.canvas.setDrawingShapeToSquare(True)

        elif event.key() in [Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4, Qt.Key_5, Qt.Key_6]:
            elements = [c for c in self.tags_group.children() if isinstance(c, QCheckBox)]
            print(f'elements: {elements}')
            for check in elements:
                if check.text().startswith(f'&{event.text()}') or check.text().startswith(f'(&{event.text().upper()})'):
                    check.setChecked(not check.isChecked())

        elif event.key() in [Qt.Key_P, Qt.Key_O, Qt.Key_I]:
            elements = [c for c in self.selection_group.children() if isinstance(c, QRadioButton)]
            for check in elements:
                if check.text().startswith(f'&{event.text().upper()}'):
                    check.setChecked(not check.isChecked())
        else:
            super().keyPressEvent(event)
        
        
    ## Support Functions ##
    def set_format(self, save_format):
        # print('save_format debug')
        if save_format == FORMAT_PASCALVOC:
            self.actions.save_format.setText(FORMAT_PASCALVOC)
            self.actions.save_format.setIcon(newIcon("format_voc"))
            self.labelFileFormat = LabelFileFormat.PASCAL_VOC
            LabelFile.suffix = XML_EXT

        elif save_format == FORMAT_XL:
            self.actions.save_format.setText(FORMAT_XL)
            self.actions.save_format.setIcon(newIcon("format_xl"))
            self.labelFileFormat = LabelFileFormat.XL
            LabelFile.suffix = TXT_EXT

        elif save_format == FORMAT_CREATEML:
            self.actions.save_format.setText(FORMAT_CREATEML)
            self.actions.save_format.setIcon(newIcon("format_createml"))
            self.labelFileFormat = LabelFileFormat.CREATE_ML
            LabelFile.suffix = JSON_EXT

        ################################### NEW STUFF ###################################
        # elif save_format == FORMAT_XL:
        #     self.actions.save_format.setText(FORMAT_XL)
        #     self.actions.save_format.setIcon(newIcon("format_xl"))
        #     self.labelFileFormat = LabelFileFormat.XL
        #     LabelFile.suffix = TXT_EXT
        ################################### NEW STUFF ####################################

    def change_format(self):
        if self.labelFileFormat == LabelFileFormat.PASCAL_VOC:
            self.set_format(FORMAT_XL)
        elif self.labelFileFormat == LabelFileFormat.XL:
            self.set_format(FORMAT_CREATEML)
        elif self.labelFileFormat == LabelFileFormat.CREATE_ML:
            self.set_format(FORMAT_PASCALVOC)
        # elif self.labelFileFormat == LabelFileFormat.XL:
        #     self.set_format(FORMAT_XL)
        else:
            raise ValueError('Unknown label file format.')
        self.setDirty()

    def noShapes(self):
        return not self.itemsToShapes

    def toggleAdvancedMode(self, value=True):
        self._beginner = not value
        self.canvas.setEditing(True)
        self.populateModeActions()
        self.editButton.setVisible(not value)
        if value:
            self.actions.createMode.setEnabled(True)
            self.actions.editMode.setEnabled(False)
            self.dock.setFeatures(self.dock.features() | self.dockFeatures)
        else:
            self.dock.setFeatures(self.dock.features() ^ self.dockFeatures)

    def populateModeActions(self):
        if self.beginner():
            tool, menu = self.actions.beginner, self.actions.beginnerContext
        else:
            tool, menu = self.actions.advanced, self.actions.advancedContext
        self.tools.clear()
        addActions(self.tools, tool)
        self.canvas.menus[0].clear()
        addActions(self.canvas.menus[0], menu)
        self.menus.edit.clear()
        actions = (self.actions.create,) if self.beginner() \
            else (self.actions.createMode, self.actions.editMode)
        addActions(self.menus.edit, actions + self.actions.editMenu)

    def setBeginner(self):
        self.tools.clear()
        addActions(self.tools, self.actions.beginner)

    def setAdvanced(self):
        self.tools.clear()
        addActions(self.tools, self.actions.advanced)

    def setDirty(self):
        self.dirty = True
        self.actions.save.setEnabled(True)

    def setClean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)
        self.actions.create.setEnabled(True)

    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.actions.zoomActions:
            z.setEnabled(value)
        for action in self.actions.onLoadActive:
            action.setEnabled(value)

    def queueEvent(self, function):
        QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def resetState(self):
        self.itemsToShapes.clear()
        self.shapesToItems.clear()
        self.labelList.clear()
        self.filePath = None
        self.imageData = None
        self.labelFile = None
        self.canvas.resetState()
        self.labelCoordinates.clear()
        self.comboBox.cb.clear()

    def currentItem(self):
        items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None

    def addRecentFile(self, filePath):
        if filePath in self.recentFiles:
            self.recentFiles.remove(filePath)
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filePath)

    def beginner(self):
        return self._beginner

    def advanced(self):
        return not self.beginner()

    def getAvailableScreencastViewer(self):
        osName = platform.system()

        if osName == 'Windows':
            return ['C:\\Program Files\\Internet Explorer\\iexplore.exe']
        elif osName == 'Linux':
            return ['xdg-open']
        elif osName == 'Darwin':
            return ['open']

    ## Callbacks ##
    def showTutorialDialog(self):
        subprocess.Popen(self.screencastViewer + [self.screencast])

    def showInfoDialog(self):
        from libs.__init__ import __version__
        msg = u'Name:{0} \nApp Version:{1} \n{2} '.format(__appname__, __version__, sys.version_info)
        QMessageBox.information(self, u'Information', msg)

    def createShape(self):
        assert self.beginner()
        self.canvas.setEditing(False)
        self.actions.create.setEnabled(False)

    def toggleDrawingSensitive(self, drawing=True):
        """In the middle of drawing, toggling between modes should be disabled."""
        self.actions.editMode.setEnabled(not drawing)
        if not drawing and self.beginner():
            # Cancel creation.
            print('Cancel creation.')
            self.canvas.setEditing(True)
            self.canvas.restoreCursor()
            self.actions.create.setEnabled(True)

    def toggleDrawMode(self, edit=True):
        self.canvas.setEditing(edit)
        self.actions.createMode.setEnabled(edit)
        self.actions.editMode.setEnabled(not edit)

    def setCreateMode(self):
        assert self.advanced()
        self.toggleDrawMode(False)

    def setEditMode(self):
        assert self.advanced()
        self.toggleDrawMode(True)
        self.labelSelectionChanged()

    def updateFileMenu(self):
        currFilePath = self.filePath

        def exists(filename):
            return os.path.exists(filename)

        menu = self.menus.recentFiles
        menu.clear()
        files = [f for f in self.recentFiles if f !=
                 currFilePath and exists(f)]
        for i, f in enumerate(files):
            icon = newIcon('labels')
            action = QAction(
                icon, '&%d %s' % (i + 1, QFileInfo(f).fileName()), self)
            action.triggered.connect(partial(self.loadRecent, f))
            menu.addAction(action)

    def popLabelListMenu(self, point):
        self.menus.labelList.exec_(self.labelList.mapToGlobal(point))

    def editLabel(self):
        if not self.canvas.editing():
            return
        item = self.currentItem()
        if not item:
            return
        text = self.labelDialog.popUp(item.text())
        if text is not None:
            item.setText(text)
            item.setBackground(generateColorByText(text))
            self.setDirty()
            self.updateComboBox()

    # Tzutalin 20160906 : Add file list and dock to move faster
    def fileitemDoubleClicked(self, item=None):
        currIndex = self.mImgList.index(ustr(item.text()))
        if currIndex < len(self.mImgList):
            filename = self.mImgList[currIndex]
            if filename:
                self.loadFile(filename)

    # Add chris
    def btnstate(self, item=None):
        """ Function to handle difficult examples
        Update on each object """
        if not self.canvas.editing():
            return

        item = self.currentItem()
        if not item:  # If not selected Item, take the first one
            item = self.labelList.item(self.labelList.count() - 1)

        difficult = self.diffcButton.isChecked()

        try:
            shape = self.itemsToShapes[item]
        except:
            pass
        # Checked and Update
        try:
            if difficult != shape.difficult:
                shape.difficult = difficult
                self.setDirty()
            else:  # User probably changed item visibility
                self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)
        except:
            pass

    # React to canvas signals.
    def shapeSelectionChanged(self, selected=False):
        if self._noSelectionSlot:
            self._noSelectionSlot = False
        else:
            shape = self.canvas.selectedShape
            if shape:
                self.shapesToItems[shape].setSelected(True)
            else:
                self.labelList.clearSelection()
        self.actions.delete.setEnabled(selected)
        self.actions.copy.setEnabled(selected)
        self.actions.edit.setEnabled(selected)
        self.actions.shapeLineColor.setEnabled(selected)
        self.actions.shapeFillColor.setEnabled(selected)

    def addLabel(self, shape):
        shape.paintLabel = self.displayLabelOption.isChecked()
        item = HashableQListWidgetItem(shape.label)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        item.setBackground(generateColorByText(shape.label))
        self.itemsToShapes[item] = shape
        self.shapesToItems[shape] = item
        self.labelList.addItem(item)
        for action in self.actions.onShapesPresent:
            action.setEnabled(True)
        self.updateComboBox()

    def remLabel(self, shape):
        if shape is None:
            # print('rm empty label')
            return
        item = self.shapesToItems[shape]
        self.labelList.takeItem(self.labelList.row(item))
        del self.shapesToItems[shape]
        del self.itemsToShapes[item]
        self.updateComboBox()

    def loadLabels(self, shapes, Transform=False, distances = None):
        s = []
        
        scroll = 0
        for label, points, line_color, fill_color, difficult in shapes:
           
                
            shape = Shape(label=label)
            for x, y in points:          
                # Ensure the labels are within the bounds of the image. If not, fix them.
                if Transform:
                    
                    warp_mat1 = np.asarray([[ 1.28808265e+00,  1.04500826e-02,  2.16956594e+02],
                                        [-8.98686458e-03,  1.27599649e+00,  3.03880377e+02]])
                    x, y = np.dot(warp_mat1, [x,y,1])
                
                
                
                x, y, snapped = self.canvas.snapPointToCanvas(x, y)
                if snapped:
                    self.setDirty()

                shape.addPoint(QPointF(x, y))
            shape.difficult = difficult
            shape.test_distance = int(distances[scroll])/1000
            shape.close()
            s.append(shape)

            if line_color:
                shape.line_color = QColor(*line_color)
            else:
                shape.line_color = generateColorByText(label)

            if fill_color:
                shape.fill_color = QColor(*fill_color)
            else:
                shape.fill_color = generateColorByText(label)

            self.addLabel(shape)
            
            scroll+=1
        self.updateComboBox()
        if self.canvas.loadShapes(s):
            self.shapes_when_file_loaded = self.canvas.loadShapes(s)
        # else:
        #     self.shapes_when_file_loaded = []
        # print(f'sh: {self.shapes_when_file_loaded}')
        ################################ NEW STUFF ################################

        # self.n_shapes = self.canvas.return_shapes(s)
        # print(f'n_shapes Load Labels: {self.n_shapes}, Length: {len(self.n_shapes)}')

        return self.shapes_when_file_loaded
    
    
    
        ################################ NEW STUFF #################################

    def updateComboBox(self):
        # Get the unique labels and add them to the Combobox.
        itemsTextList = [str(self.labelList.item(i).text()) for i in range(self.labelList.count())]

        uniqueTextList = list(set(itemsTextList))
        # Add a null row for showing all the labels
        uniqueTextList.append("")
        uniqueTextList.sort()

        self.comboBox.update_items(uniqueTextList)

    def saveLabels(self, annotationFilePath):
        annotationFilePath = ustr(annotationFilePath)
        if self.labelFile is None:
            self.labelFile = LabelFile()
            self.labelFile.verified = self.canvas.verified

        def format_shape(s):
            return dict(label=s.label,
                        line_color=s.line_color.getRgb(),
                        fill_color=s.fill_color.getRgb(),
                        points=[(p.x(), p.y()) for p in s.points],
                        # add chris
                        difficult=s.difficult)

        shapes = [format_shape(shape) for shape in self.canvas.shapes]
        # Can add differrent annotation formats here
        try:
            if self.labelFileFormat == LabelFileFormat.PASCAL_VOC:
                if annotationFilePath[-4:].lower() != ".xml":
                    annotationFilePath += XML_EXT
                self.labelFile.savePascalVocFormat(annotationFilePath, shapes, self.filePath, self.imageData,
                                                   self.lineColor.getRgb(), self.fillColor.getRgb())

            elif self.labelFileFormat == LabelFileFormat.XL:
                if annotationFilePath[-4:].lower() != ".txt":
                    annotationFilePath += TXT_EXT
                self.labelFile.saveXLFormat(annotationFilePath, shapes, self.filePath, self.imageData, self.labelHist,
                                            self.lineColor.getRgb(), self.fillColor.getRgb())

            elif self.labelFileFormat == LabelFileFormat.CREATE_ML:
                if annotationFilePath[-5:].lower() != ".json":
                    annotationFilePath += JSON_EXT
                self.labelFile.saveCreateMLFormat(annotationFilePath, shapes, self.filePath, self.imageData,
                                                  self.labelHist, self.lineColor.getRgb(), self.fillColor.getRgb())

            else:
                self.labelFile.save(annotationFilePath, shapes, self.filePath, self.imageData,
                                    self.lineColor.getRgb(), self.fillColor.getRgb())
            print('Image:{0} -> Annotation:{1}'.format(self.filePath, annotationFilePath))

            return True

        except LabelFileError as e:
            self.errorMessage(u'Error saving label data', u'<b>%s</b>' % e)
            return False

    def copySelectedShape(self):
        self.addLabel(self.canvas.copySelectedShape())
        # fix copy and delete
        self.shapeSelectionChanged(True)

    def comboSelectionChanged(self, index):
        text = self.comboBox.cb.itemText(index)
        for i in range(self.labelList.count()):
            if text == "":
                self.labelList.item(i).setCheckState(2)
            elif text != self.labelList.item(i).text():
                self.labelList.item(i).setCheckState(0)
            else:
                self.labelList.item(i).setCheckState(2)

    def labelSelectionChanged(self):
        item = self.currentItem()
        if item and self.canvas.editing():
            # print('I am in selectionChanged')
            self._noSelectionSlot = True
            self.canvas.selectShape(self.itemsToShapes[item])
            shape = self.itemsToShapes[item]
            # Add Chris
            self.diffcButton.setChecked(shape.difficult)

    def labelItemChanged(self, item):
        shape = self.itemsToShapes[item]
        label = item.text()
        if label != shape.label:
            shape.label = item.text()
            shape.line_color = generateColorByText(shape.label)
            self.setDirty()
        else:  # User probably changed item visibility
            self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)

    # Callback functions:
    def newShape(self):
        """Pop-up and give focus to the label editor.
        position MUST be in global coordinates.
        """
        if not self.useDefaultLabelCheckbox.isChecked() or not self.defaultLabelTextLine.text():
            if len(self.labelHist) > 0:
                self.labelDialog = LabelDialog(
                    parent=self, listItem=self.labelHist)

            # Sync single class mode from PR#106
            if self.singleClassMode.isChecked() and self.lastLabel:
                text = self.lastLabel
            else:
                text = self.labelDialog.popUp(text=self.prevLabelText)
                self.lastLabel = text
        else:
            text = self.defaultLabelTextLine.text()

        # Add Chris
        self.diffcButton.setChecked(False)
        if text is not None:
            self.prevLabelText = text
            generate_color = generateColorByText(text)
            shape = self.canvas.setLastLabel(text, generate_color, generate_color)
            self.addLabel(shape)
            if self.beginner():  # Switch to edit mode.
                self.canvas.setEditing(True)
                self.actions.create.setEnabled(True)
            else:
                self.actions.editMode.setEnabled(True)
            self.setDirty()

            if text not in self.labelHist:
                self.labelHist.append(text)
        else:
            # self.canvas.undoLastLine()
            self.canvas.resetAllLines()

    def scrollRequest(self, delta, orientation):
        units = - delta / (8 * 15)
        bar = self.scrollBars[orientation]
        bar.setValue(bar.value() + bar.singleStep() * units)

    def setZoom(self, value):
        self.actions.fitWidth.setChecked(False)
        self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)

    def addZoom(self, increment=10):
        self.setZoom(self.zoomWidget.value() + increment)

    def zoomRequest(self, delta):
        # get the current scrollbar positions
        # calculate the percentages ~ coordinates
        h_bar = self.scrollBars[Qt.Horizontal]
        v_bar = self.scrollBars[Qt.Vertical]

        # get the current maximum, to know the difference after zooming
        h_bar_max = h_bar.maximum()
        v_bar_max = v_bar.maximum()

        # get the cursor position and canvas size
        # calculate the desired movement from 0 to 1
        # where 0 = move left
        #       1 = move right
        # up and down analogous
        cursor = QCursor()
        pos = cursor.pos()
        relative_pos = QWidget.mapFromGlobal(self, pos)

        cursor_x = relative_pos.x()
        cursor_y = relative_pos.y()

        w = self.scrollArea.width()
        h = self.scrollArea.height()

        # the scaling from 0 to 1 has some padding
        # you don't have to hit the very leftmost pixel for a maximum-left movement
        margin = 0.1
        move_x = (cursor_x - margin * w) / (w - 2 * margin * w)
        move_y = (cursor_y - margin * h) / (h - 2 * margin * h)

        # clamp the values from 0 to 1
        move_x = min(max(move_x, 0), 1)
        move_y = min(max(move_y, 0), 1)

        # zoom in
        units = delta / (8 * 15)
        scale = 10
        self.addZoom(scale * units)

        # get the difference in scrollbar values
        # this is how far we can move
        d_h_bar_max = h_bar.maximum() - h_bar_max
        d_v_bar_max = v_bar.maximum() - v_bar_max

        # get the new scrollbar values
        new_h_bar_value = h_bar.value() + move_x * d_h_bar_max
        new_v_bar_value = v_bar.value() + move_y * d_v_bar_max

        h_bar.setValue(new_h_bar_value)
        v_bar.setValue(new_v_bar_value)

    def setFitWindow(self, value=True):
        if value:
            self.actions.fitWidth.setChecked(False)
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def togglePolygons(self, value):
        for item, shape in self.itemsToShapes.items():
            item.setCheckState(Qt.Checked if value else Qt.Unchecked)

    def loadFile(self, filePath=None):
        """Load the specified file, or the last opened file if None."""
        self.resetState()
        self.canvas.setEnabled(False)
        if filePath is None:
            filePath = self.settings.get(SETTING_FILENAME)

        # Make sure that filePath is a regular python string, rather than QString
        filePath = ustr(filePath)
        # print(f'filePath: {filePath}')

        # Fix bug: An  index error after select a directory when open a new file.
        unicodeFilePath = ustr(filePath)
        # print(f'unicodeFilePath: {unicodeFilePath}')

        unicodeFilePath = os.path.abspath(unicodeFilePath)
        # print(f'Absolute unicodeFilePath: {unicodeFilePath}')

        # Tzutalin 20160906 : Add file list and dock to move faster
        # Highlight the file item
        if unicodeFilePath and self.fileListWidget.count() > 0:
            if unicodeFilePath in self.mImgList:
                index = self.mImgList.index(unicodeFilePath)
                fileWidgetItem = self.fileListWidget.item(index)
                fileWidgetItem.setSelected(True)
            else:
                self.fileListWidget.clear()
                self.mImgList.clear()
        
            
        if unicodeFilePath and os.path.exists(unicodeFilePath):
            if LabelFile.isLabelFile(unicodeFilePath):
                try:
                    self.labelFile = LabelFile(unicodeFilePath)
                    print("============", unicodeFilePath)
                except LabelFileError as e:
                    self.errorMessage(u'Error opening file',
                                      (u"<p><b>%s</b></p>"
                                       u"<p>Make sure <i>%s</i> is a valid label file.")
                                      % (e, unicodeFilePath))
                    self.status("Error reading %s" % unicodeFilePath)
                    return False
                self.imageData = self.labelFile.imageData
                # print('imageData: '.format(self.imageData))
                self.lineColor = QColor(*self.labelFile.lineColor)
                self.fillColor = QColor(*self.labelFile.fillColor)
                self.canvas.verified = self.labelFile.verified

            ############################################## NEW STUFF #####################################
            elif unicodeFilePath.endswith('bin') or unicodeFilePath.endswith('vis'):
                print("============", unicodeFilePath)
                
                img = self.read_matrix(unicodeFilePath)
                print(img.shape)
                ceil = np.percentile(img, int(self.ceil_le.text()))
                floor = np.percentile(img, int(self.floor_le.text()))
                out = scale_image_base(img, ceil, floor)
                copy = out.copy()
                image = numpyQImage(copy)
                self.imageData = image
            ############################################## NEW STUFF ######################################

            else:
                # Load image:
                # read data first and store for saving into label file.
                print(f"End: {unicodeFilePath.endswith('jpg')}")
                self.imageData = read(unicodeFilePath, None)
                # print('imageData: '.format(self.imageData))
                self.labelFile = None
                self.canvas.verified = False

            if isinstance(self.imageData, QImage):
                image = self.imageData
            else:
                image = QImage.fromData(self.imageData)
            if image.isNull():
                self.errorMessage(u'Error opening file',
                                  u"<p>Make sure <i>%s</i> is a valid image file." % unicodeFilePath)
                self.status("Error reading %s" % unicodeFilePath)
                return False
            self.status("Loaded %s" % os.path.basename(unicodeFilePath))
            self.image = image
            self.filePath = unicodeFilePath
            self.canvas.loadPixmap(QPixmap.fromImage(image))
            if self.labelFile:
                self.loadLabels(self.labelFile.shapes)
            self.setClean()
            self.canvas.setEnabled(True)
            self.adjustScale(initial=True)
            self.paintCanvas()
            self.addRecentFile(self.filePath)
            self.toggleActions(True)
            self.showBoundingBoxFromAnnotationFile(filePath, Transform=False)

            # print(f'loadFile shapes: {self.n_shapes}')

            self.setWindowTitle(__appname__ + ' ' + filePath)

            # Default : select last item if there is at least one item
            if self.labelList.count():
                self.labelList.setCurrentItem(self.labelList.item(self.labelList.count() - 1))
                self.labelList.item(self.labelList.count() - 1).setSelected(True)

            self.canvas.setFocus(True)
            return True
        
            
        return False
    
    
    def extractBaseName(self, filePath):
        
        baseName = os.path.splitext(filePath)[0].split('/')[-1].replace('xl_visual','')
        
        return baseName

    def showBoundingBoxFromAnnotationFile(self, filePath, Transform=False):
        # prefix i.e. session_name
        # prefix = 'session' + '_' + '_'.join(self.session_name.split('_')[:2])
        prefix = 'session' + '_' + self.session_name
        print(f'DefaultSaveDir: {self.defaultSaveDir}')
        read_OD_annots = os.path.join(self.defaultSaveDir.split('VisualImages_annots')[0], 'Annotation_OD_refined')
        
        '"/media/adarsh/USB DRIVE/Ground truth generation/test/Images/Annotation_OD_refined/'
        annotation_filePath = os.path.join (self.AnnotationRootDir ,'xl_visual' + self.extractBaseName(filePath) + '.txt')
        
        print('888888888888888', annotation_filePath)
        if self.defaultSaveDir is not None:

            if self.task_type == 'Object_Detection':
                self.read_from_annotation_OD = os.path.join(self.defaultSaveDir.split('VisualImages_annots')[0],
                                                            'Annotation_OD_refined')
                print(f'read_from_annotation_OD: {self.read_from_annotation_OD}')
            elif self.task_type == 'Lane_Detection':
                self.read_from_annotation_LD = os.path.join(self.defaultSaveDir.split('Annotation_LD_refined')[0],
                                                            'Annotation_LD')

            # basename = os.path.basename(os.path.splitext(filePath)[0])
            basename = os.path.basename(os.path.splitext(filePath)[0]).split('_')
            print(f'BaseName: {basename}')
            # new_base_name = basename[0] + '_' + basename[1] + '_' + 'od' + '_' + basename[-1]
            new_base_name = '_'.join(idx for idx in basename)
            print(f'New Base name: {new_base_name}')
            filedir = filePath.split(new_base_name)[0].split(os.path.sep)[-2:-1][0]
            print(f'filedir: {filedir}')
            xmlPath = os.path.join(self.defaultSaveDir, new_base_name + XML_EXT)
            # txtPath = os.path.join(self.read_from_annotation_OD, new_base_name + TXT_EXT).replace('\\', '/')
            txtPath = os.path.join(self.defaultSaveDir, new_base_name + TXT_EXT).replace('\\', '/')
            print(f'txtPath: {txtPath}')
            refined_dir = os.path.join(Path(txtPath.split(os.path.basename(txtPath))[0]).parents[0],
                                       'Annotation_OD_refined')
            #H:/AI_Dataset/XLG3_Toyo_Panasonic_Recording_2021_08/ToyoPanasonicRecording_2021-08-10_10.53.41/AI_Data
            #C:\Thesis\Dataset\Toyo
            
            #C:/Thesis/Dataset/Toyo/AI_Dataset/Annotation_OD_refined   
            #G:/Ground truth generation/test/Images/Annotation_OD_refined
            #/media/adarsh/USB DRIVE/Ground truth generation/test/Images/Annotation_OD_refined
            
            txtPath = "/media/adarsh/USB DRIVE/Ground truth generation/test/Images/Annotation_OD_refined/xl_visual" + txtPath.split("/")[-1].split('xl_visual')[-1]
            
            txtPath = annotation_filePath
            
            refined_txtPath = txtPath.replace("Annotation_OD", "Annotation_OD_refined")
            
            jsonPath = os.path.join(self.defaultSaveDir, filedir + JSON_EXT)

            # Activate the "Save" Button once the file is loaded
            self.actions.save.setEnabled(True)

            """Annotation file priority:
            XL > CreateML > PascalXML
            """
            if os.path.isfile(xmlPath):
                self.loadPascalXMLByFilename(xmlPath)
            elif os.path.isfile(txtPath) or os.path.isfile(refined_txtPath):
                if os.path.basename(txtPath) in os.listdir(refined_dir):
                    # self.loadXLTXTByFilename(refined_txtPath)
                    self.loadXLTXTByFilename(txtPath, Transform)
                    
                else:
                    self.loadXLTXTByFilename(txtPath)
            elif os.path.isfile(jsonPath):
                self.loadCreateMLJSONByFilename(jsonPath, filePath)

        else:
            xmlPath = os.path.splitext(filePath)[0] + XML_EXT
            txtPath = os.path.splitext(filePath)[0] + TXT_EXT
            # print(f'In else txtPath: {txtPath}')
            if os.path.isfile(xmlPath):
                self.loadPascalXMLByFilename(xmlPath)
            elif os.path.isfile(txtPath):
                self.loadXLTXTByFilename(txtPath)
            # elif os.path.isfile(txtPath) and self.format == 'XL':
            #     print('else debug')
            #     self.loadXLTXTByFilename(txtPath)

    def resizeEvent(self, event):
        if self.canvas and not self.image.isNull() \
                and self.zoomMode != self.MANUAL_ZOOM:
            self.adjustScale()
        super(MainWindow, self).resizeEvent(event)

    def paintCanvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.labelFontSize = int(0.02 * max(self.image.width(), self.image.height()))
        self.canvas.adjustSize()
        self.canvas.update()

    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        self.zoomWidget.setValue(int(100 * value))

    def scaleFitWindow(self):
        """Figure out the size of the pixmap in order to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e
        h1 = self.centralWidget().height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0
        return w / self.canvas.pixmap.width()

    def closeEvent(self, event):
        if not self.mayContinue():
            event.ignore()
        settings = self.settings
        # If it loads images from dir, don't load it at the begining
        if self.dirname is None:
            settings[SETTING_FILENAME] = self.filePath if self.filePath else ''
        else:
            settings[SETTING_FILENAME] = ''

        settings[SETTING_WIN_SIZE] = self.size()
        settings[SETTING_WIN_POSE] = self.pos()
        settings[SETTING_WIN_STATE] = self.saveState()
        settings[SETTING_LINE_COLOR] = self.lineColor
        settings[SETTING_FILL_COLOR] = self.fillColor
        settings[SETTING_RECENT_FILES] = self.recentFiles
        settings[SETTING_ADVANCE_MODE] = not self._beginner
        if self.defaultSaveDir and os.path.exists(self.defaultSaveDir):
            settings[SETTING_SAVE_DIR] = ustr(self.defaultSaveDir)
        else:
            settings[SETTING_SAVE_DIR] = ''

        if self.lastOpenDir and os.path.exists(self.lastOpenDir):
            settings[SETTING_LAST_OPEN_DIR] = self.lastOpenDir
        else:
            settings[SETTING_LAST_OPEN_DIR] = ''

        settings[SETTING_AUTO_SAVE] = self.autoSaving.isChecked()
        settings[SETTING_SINGLE_CLASS] = self.singleClassMode.isChecked()
        settings[SETTING_PAINT_LABEL] = self.displayLabelOption.isChecked()
        settings[SETTING_DRAW_SQUARE] = self.drawSquaresOption.isChecked()
        settings[SETTING_LABEL_FILE_FORMAT] = self.labelFileFormat
        settings.save()

    def loadRecent(self, filename):
        if self.mayContinue():
            self.loadFile(filename)

    def scanAllImages(self, folderPath):
        extensions = ['.%s' % fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
        ############################### NEW STUFF #################################
        extensions.append('.bin')
        extensions.append('.vis')
        extensions.remove('.ppm')
        
        
        ################################ NEW STUFF #################################
        images = []

        for root, dirs, files in os.walk(folderPath):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    relativePath = os.path.join(root, file)
                    path = ustr(os.path.abspath(relativePath))
                    images.append(path)
        natural_sort(images, key=lambda x: x.lower())

        ############################### NEW STUFF ###########################
        # Progress Bar
        self.progress.setMaximum(len(images)-1)
        ############################### NEW STUFF ###########################

        return images

    def changeSavedirDialog(self, _value=False):
        if self.defaultSaveDir is not None:
            path = ustr(self.defaultSaveDir)
        else:
            path = '.'

        dirpath = ustr(QFileDialog.getExistingDirectory(self,
                                                        '%s - Save annotations to the directory' % __appname__, path,
                                                        QFileDialog.ShowDirsOnly
                                                         | QFileDialog.DontResolveSymlinks))
        print(f'changedirdialog: {dirpath}')

        if dirpath is not None and len(dirpath) > 1:
            self.defaultSaveDir = dirpath

        self.statusBar().showMessage('%s . Annotation will be saved to %s' %
                                     ('Change saved folder', self.defaultSaveDir))
        self.statusBar().show()

    def openAnnotationDialog(self, _value=False):
        if self.filePath is None:
            self.statusBar().showMessage('Please select image first')
            self.statusBar().show()
            return

        path = os.path.dirname(ustr(self.filePath)) \
            if self.filePath else '.'
        if self.labelFileFormat == LabelFileFormat.PASCAL_VOC:
            filters = "Open Annotation XML file (%s)" % ' '.join(['*.xml'])
            filename = ustr(QFileDialog.getOpenFileName(self, '%s - Choose a xml file' % __appname__, path, filters))
            if filename:
                if isinstance(filename, (tuple, list)):
                    filename = filename[0]
            self.loadPascalXMLByFilename(filename)

    def openDirDialog(self, _value=False, dirpath=None, silent=False, recursive=False):

        ################################# NEW STUFF ######################################
        # Call the make annotation directories function
        if self.task_type == 'Object_Detection':
            self.annotation_dir = self.make_object_detection_annotation_directory()
            # print(f'Annotation_Dir: {self.annotation_dir}')
        elif self.task_type == 'Lane_Detection':
            self.annotation_dir = self.make_lane_detection_annotation_directory()
        ################################# NEW STUFF ######################################

        if not self.mayContinue():
            return

        defaultOpenDirPath = dirpath if dirpath else '.'

        if self.lastOpenDir and os.path.exists(self.lastOpenDir):
            defaultOpenDirPath = self.lastOpenDir
        else:
            defaultOpenDirPath = os.path.dirname(self.filePath) if self.filePath else '.'
        if not silent:
            targetDirPath = ustr(QFileDialog.getExistingDirectory(self,
                                                                  '%s - Open Directory' % __appname__,
                                                                  defaultOpenDirPath,
                                                                  QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))
            ################# NEW STUFF ################
            # if not self.defaultSaveDir:
            # self.defaultSaveDir = targetDirPath
            print(f'targetDirPath: {targetDirPath}')
            self.session_name = os.path.basename(targetDirPath)
            print(f'session_name: {self.session_name}')
            # targetDirPathBasename = os.path.basename(targetDirPath)
            # new_targetDirPath = targetDirPath + r'/' + os.path.basename(targetDirPath) + '_' + 'annots'
            # self.defaultSaveDir = new_targetDirPath
            # print(f'OpenDirDialog newtargetDirPath: {new_targetDirPath}')
            self.defaultSaveDir = self.annotation_dir  # Make 'Annotation_OD' dir default directory for saving
            # self.defaultSaveDir = new_targetDirPath
            ################# NEW STUFF ################
        else:
            targetDirPath = ustr(defaultOpenDirPath)

        self.lastOpenDir = targetDirPath
        self.importDirImages(targetDirPath)  # old


    def importDirImages(self, dirpath, recursive=False):
        # print(f'last: {self.lastOpenDir}')
        """
        ####################################### NEW STUFF ############################

        Added:
            param: recursive
            variable: self.database
        ####################################### NEW STUFF ############################
        """
        if not self.mayContinue() or not dirpath:
            return

        self.lastOpenDir = dirpath
        self.dirname = dirpath
        # print(f'self.dirname: {self.dirname}')
        ######################################### NEW STUFF #############################
        # if self.dirname:
        self.database = Database(Path(self.dirname), recursive)
        # ######################################### NEW STUFF #############################

        ###################### NOT USEFUL ##########################
        # self.filePath = None
        # self.fileListWidget.clear()
        # self.mImgList = self.scanAllImages(dirpath)
        # self.openNextImg()
        # for imgPath in self.mImgList:
        #     # print(f'imgPath: {imgPath}')
        #     item = QListWidgetItem(imgPath)
        #     self.fileListWidget.addItem(item)
        # self.fetchDatabase()
        # print(f'Import Dir Images filepath: {self.filePath}')

        ########################################## NEW STUFF ###############################
        self.filePath = None
        self.fileListWidget.clear()
        self.mImgList = self.scanAllImages(dirpath)
        self.openNextImg()
        for imgPath in self.mImgList:
            item = QListWidgetItem(imgPath)
            self.fileListWidget.addItem(item)
        
        # self.fetchDatabase()
        # print(f'Import Dir Images filepath: {self.filePath}')
        ########################################## NEW STUFF ###############################

    def verifyImg(self, _value=False):
        # Proceding next image without dialog if having any label
        if self.filePath is not None:
            try:
                self.labelFile.toggleVerify()
            except AttributeError:
                # If the labelling file does not exist yet, create if and
                # re-save it with the verified attribute.
                self.saveFile()
                if self.labelFile != None:
                    self.labelFile.toggleVerify()
                else:
                    return

            self.canvas.verified = self.labelFile.verified
            self.paintCanvas()
            self.saveFile()

    def openPrevImg(self, _value=False):
        self.pointCloud.setChecked(False)
        
        ############################ NEW STUFF #####################
        # self.image_idx = self.image_idx - 1
        ############################ NEW STUFF #####################

        # Proceding prev image without dialog if having any label
        if self.autoSaving.isChecked():
            if self.defaultSaveDir is not None:
                if self.dirty is True:
                    self.saveFile()
            else:
                self.changeSavedirDialog()
                return

        if not self.mayContinue():
            return

        if len(self.mImgList) <= 0:
            return

        if self.filePath is None:
            return

        currIndex = self.mImgList.index(self.filePath)
        ##################### NEW STUFF ##############################
        self.image_idx = currIndex-1
        if self.image_idx < 0:
            self.image_idx = 0
        self.image_index_textbox.setText(str(self.image_idx))
        self.progress.setValue(currIndex-1)
        ##################### NEW STUFF ##############################
        if currIndex - 1 >= 0:
            filename = self.mImgList[currIndex - 1]

            ###################### NEW STUFF ##################
            self.updateDatabase()
            ###################### NEW STUFF ##################
            if filename:
                print(f'openNext filename: {filename}')
                print('-----------------------------------------------')
                self.loadFile(filename)
                ###################### NEW STUFF ##################
                self.fetchDatabase()

                if not self.canvas.shapes:
                    self.shapeCountBefore = 0
                else:
                    self.shapeCountBefore = len(self.canvas.shapes)
                print(f'openPrevious: {self.shapeCountBefore}')
                ###################### NEW STUFF ##################

    def openNextImg(self, _value=False):
        # Proceding prev image without dialog if having any label
        self.pointCloud.setChecked(False)
        if self.autoSaving.isChecked():
            if self.defaultSaveDir is not None:
                if self.dirty is True:
                    self.saveFile()
            else:
                self.changeSavedirDialog()
                return

        if not self.mayContinue():
            return

        if len(self.mImgList) <= 0:
            return

        filename = None
        if self.filePath is None:
            #print('Inside here')
            filename = self.mImgList[0]
            ##################### NEW STUFF ##############################
            self.image_idx = 0
            self.image_index_textbox.setText(str(self.image_idx))
            # self.saveFile()  # NEW STUFF
            ##################### NEW STUFF ##############################
        else:
            #print('No inside here')
            currIndex = self.mImgList.index(self.filePath)
            # self.saveFile()  # NEW STUFF
            ##################### NEW STUFF ##############################
            self.image_idx = currIndex+1
            if self.image_idx >= len(self.mImgList):
                self.image_idx = currIndex
            self.image_index_textbox.setText(str(self.image_idx))
            self.progress.setValue(currIndex+1)
            ##################### NEW STUFF ##############################
            self.updateDatabase()
            if currIndex + 1 < len(self.mImgList):
                filename = self.mImgList[currIndex + 1]
            # print(f'Update Database Filepath: {self.filePath}')
        ###################### NEW STUFF ##################
        # self.updateDatabase()
        ###################### NEW STUFF ##################
        if filename:
            #print('Nope here')
            self.loadFile(filename)
            self.updateDatabase()
            # self.saveFile()  # New Stuff
            ###################### NEW STUFF ##################
            self.fetchDatabase()

            if not self.canvas.shapes:
                self.shapeCountBefore = 0
            else:
                self.shapeCountBefore = len(self.canvas.shapes)
            # print(f'openNext: {self.shapeCountBefore}')
            ###################### NEW STUFF ##################

        ######################################### NEW STUFF #################################

    def make_object_detection_annotation_directory(self, name_dir='Annotation_OD_refined'):
        print(f'last Open: {self.lastOpenDir}')
        if os.path.exists(self.lastOpenDir):
            print("last opened dir ", self.lastOpenDir)
        else:
            print("last opened dir is no more available")
            
        basedir = os.path.join(self.lastOpenDir, os.path.basename(self.lastOpenDir) + '_' + 'annots').replace('\\', '/')
        object_detection_annotation_target_dir = os.path.join(self.lastOpenDir, name_dir).replace("/", "\\")
        if os.path.exists(object_detection_annotation_target_dir):
            pass
        else:
            os.makedirs(object_detection_annotation_target_dir)
        
        return object_detection_annotation_target_dir

    def make_lane_detection_annotation_directory(self, name_dir='Annotation_LD_refined'):
        basedir = os.path.join(self.lastOpenDir, os.path.basename(self.lastOpenDir) + '_' + 'annots').replace('\\', '/')
        lane_detection_annotation_target_dir = os.path.join(basedir, name_dir).replace("/", "\\")
        if os.path.exists(lane_detection_annotation_target_dir):
            pass
        else:
            os.makedirs(lane_detection_annotation_target_dir)
        return lane_detection_annotation_target_dir

    def read_matrix(self, file_name):
        # data = np.fromfile(file_name, dtype='<d')  # little-endian double precision float
        # nr_rows = 512
        # nr_cols = int(len(data) / nr_rows)
        # img = data.reshape((nr_rows, nr_cols))
        c, m = xw.XW_ReadFile(file_name)
        img = c['data']
        return img

    @pyqtSlot()
    def image_index_textbox_pressed(self):
        inp_text = self.image_index_textbox.text()
        try:
            input_text = int(inp_text)
        except Exception as e:
            input_text = self.image_idx
        self.updateDatabase()
        self.image_idx = max(0, min(len(self.mImgList) - 1, input_text))
        self.fetchDatabase()
        self.loadFile(self.mImgList[self.image_idx])
        self.image_index_textbox.setText(str(self.image_idx))

    def progress_bar(self):
        control_dock = QDockWidget('Navigation', self)
        self.next_button = QPushButton('&Next')
        self.prev_button = QPushButton('&Previous')
        self.image_index_textbox = QLineEdit(self)
        self.progress = QProgressBar(self)
        # self.progress.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.next_button.clicked.connect(self.openNextImg)
        self.prev_button.clicked.connect(self.openPrevImg)
        self.image_index_textbox.returnPressed.connect(self.image_index_textbox_pressed)

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(self.prev_button)
        hbox.addWidget(self.image_index_textbox)
        hbox.addWidget(self.next_button)

        vbox = QVBoxLayout()
        vbox.addStretch(1)
        vbox.addWidget(self.progress)
        vbox.addLayout(hbox)

        buttons_widget = QWidget()
        buttons_widget.setLayout(vbox)
        control_dock.setWidget(buttons_widget)
        control_dock.setFloating(False)
        control_dock.widget().setMinimumSize(QSize(self.min_dock_width, 90))
        self.addDockWidget(Qt.RightDockWidgetArea, control_dock)
        # pass
        
    def progress_bar_RGB(self):
        
        control_dock = QDockWidget('Input selection', self)
        self.RGB_button = QPushButton('&RGB')
        self.VSCEL_button = QPushButton('&Visual')
        self.Fuse_button = QPushButton('&Fuse')
        self.pointCloud = QCheckBox('&Pointcloud')
        
        self.pointCloud.setChecked(False)
        self.pointCloud.stateChanged.connect(self.add_point_cloud)
        
        #self.image_index_textbox = QLineEdit(self)
        #self.progress = QProgressBar(self)
        # self.progress.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.RGB_button.clicked.connect(self.openNextRGB)
        self.VSCEL_button.clicked.connect(self.openNextVisual)
        self.Fuse_button.clicked.connect(self.Fuse_RGB_Visual)
        
        #self.image_index_textbox.returnPressed.connect(self.image_index_textbox_pressed)

        hbox = QHBoxLayout()
        #hbox.addStretch(1)
        hbox.addWidget(self.VSCEL_button)
        #hbox.addWidget(self.image_index_textbox)
        hbox.addWidget(self.RGB_button)
        hbox.addWidget(self.Fuse_button)
        hbox.addWidget(self.pointCloud)

        vbox = QVBoxLayout()
        #vbox.addStretch(1)
        #vbox.addWidget(self.progress)
        vbox.addLayout(hbox)

        buttons_widget = QWidget()
        buttons_widget.setLayout(vbox)
        control_dock.setWidget(buttons_widget)
        control_dock.setFloating(False)
        control_dock.widget().setMinimumSize(QSize(self.min_dock_width, 90))
        self.addDockWidget(Qt.RightDockWidgetArea, control_dock)
    
    def read_pointCloud(self,filepath):
        pnt_ = self.read_matrix(filepath)
        points=[]
        for i in range(pnt_.shape[0]):
             
            x = pnt_[:,4:6][i][0]
            y = pnt_[:,4:6][i][1]

            if x<0 or y<0:
                pass
            else:
                points.append((x,y))
                
        return points
    
    def add_point_cloud(self):
        
        if self.pointCloud.isChecked():
            print("display pointcloud")
            self.canvas.show_PTcloud= True
            point_cloud_file = self.filePath.replace(".vis",".xpc").replace("xl_visual", "xw_pointcloud")
            points = self.read_pointCloud(point_cloud_file)
            self.canvas.points = points
            print("Number of points ", len(points))
            
                
        else:
            self.canvas.show_PTcloud= False
            print("pointcloud off")
        self.canvas.repaint()
    
    def openNextRGB(self, _value=False):
        # Proceding prev image without dialog if having any label
        
        
        rgb_path = self.filePath.replace(".vis", ".ppm").replace("xl_visual","xl_rgbImage")
        #print("its here testing RGB thing", rgb_path)
        #print(self.mImgList)
        self.loadFile_RGB(filePath=rgb_path)
        #print("Check ======================>", self.mImgList)
        
    def Fuse_RGB_Visual(self, _value=False):
        
        rgb_path = self.filePath.replace(".vis", ".ppm").replace("xl_visual","xl_rgbImage")
        print(rgb_path)
        print(self.filePath)
        self.loadFile_RGB(filePath=rgb_path, Visual_path=self.filePath, Fuse=True)
        
        
    def loadFile_RGB(self, filePath=None,Visual_path=None, Fuse=False):
        
        
        self.resetState()
        self.canvas.setEnabled(False)
        if filePath is None:
            filePath = self.settings.get(SETTING_FILENAME)

        # Make sure that filePath is a regular python string, rather than QString
        filePath = ustr(filePath)
        # print(f'filePath: {filePath}')

        # Fix bug: An  index error after select a directory when open a new file.
        unicodeFilePath = ustr(filePath)
        # print(f'unicodeFilePath: {unicodeFilePath}')

        unicodeFilePath = os.path.abspath(unicodeFilePath)
        # print(f'Absolute unicodeFilePath: {unicodeFilePath}')

        # Tzutalin 20160906 : Add file list and dock to move faster
        # Highlight the file item
        '''
        if unicodeFilePath and self.fileListWidget.count() > 0:
            if unicodeFilePath in self.mImgList:
                index = self.mImgList.index(unicodeFilePath)
                fileWidgetItem = self.fileListWidget.item(index)
                fileWidgetItem.setSelected(True)
            else:
                self.fileListWidget.clear()
                self.mImgList.clear()
        '''
        if unicodeFilePath and os.path.exists(unicodeFilePath):
            if LabelFile.isLabelFile(unicodeFilePath):
                try:
                    self.labelFile = LabelFile(unicodeFilePath)
                except LabelFileError as e:
                    self.errorMessage(u'Error opening file',
                                      (u"<p><b>%s</b></p>"
                                       u"<p>Make sure <i>%s</i> is a valid label file.")
                                      % (e, unicodeFilePath))
                    self.status("Error reading %s" % unicodeFilePath)
                    return False
                self.imageData = self.labelFile.imageData
                # print('imageData: '.format(self.imageData))
                self.lineColor = QColor(*self.labelFile.lineColor)
                self.fillColor = QColor(*self.labelFile.fillColor)
                self.canvas.verified = self.labelFile.verified

           
            elif unicodeFilePath.endswith('ppm'):
                # Load image:
                # read data first and store for saving into label file.
                print(f"End: {unicodeFilePath.endswith('ppm')}")
                img = cv2.imread(unicodeFilePath)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                print(">>>>>>", img.shape)
                
                ## fusion to be added here
                if Fuse:
                    
                    warp_mat1 = np.asarray([[ 7.87257477e-01,  3.03357919e-02, -2.01224628e+02],
                                            [ 6.12790582e-03,  7.87165388e-01, -2.38532559e+02]])
                    img = cv2.warpAffine(img, warp_mat1, (1536, 512))
                    
                image = numpyQImage(img)
                self.imageData = image
                # print('imageData: '.format(self.imageData))
                #self.labelFile = None
                #self.canvas.verified = False
            
            
            else:
                # Load image:
                # read data first and store for saving into label file.
                print(f"End: {unicodeFilePath.endswith('jpg')}")
                self.imageData = read(unicodeFilePath, None)
                # print('imageData: '.format(self.imageData))
                self.labelFile = None
                self.canvas.verified = False

            if isinstance(self.imageData, QImage):
                image = self.imageData
            else:
                image = QImage.fromData(self.imageData)
            if image.isNull():
                self.errorMessage(u'Error opening file',
                                  u"<p>Make sure <i>%s</i> is a valid image file." % unicodeFilePath)
                self.status("Error reading %s" % unicodeFilePath)
                return False
            
            
            
            #print("image_shape is here ++++", self.QImageToCvMat(image).shape)
            self.status("Loaded %s" % os.path.basename(unicodeFilePath))
            self.image = image
            self.filePath = unicodeFilePath.replace(".ppm",".vis").replace("xl_rgbImage","xl_visual")
            
            self.canvas.loadPixmap(QPixmap.fromImage(image))
            if self.labelFile:
                
                if Fuse:
                    self.loadLabels(self.labelFile.shapes, Transform=False)
                else:
                    self.loadLabels(self.labelFile.shapes, Transform=True)
            
            self.setClean()
            self.canvas.setEnabled(True)
            self.adjustScale(initial=True)
            self.paintCanvas()
            self.addRecentFile(self.filePath)
            self.toggleActions(True)
            if Fuse:
                self.showBoundingBoxFromAnnotationFile(self.filePath, Transform=False)
            else:
                
                self.showBoundingBoxFromAnnotationFile(self.filePath, Transform=True)
            

            # print(f'loadFile shapes: {self.n_shapes}')

            self.setWindowTitle(__appname__ + ' ' + filePath)

            # Default : select last item if there is at least one item
            if self.labelList.count():
                self.labelList.setCurrentItem(self.labelList.item(self.labelList.count() - 1))
                self.labelList.item(self.labelList.count() - 1).setSelected(True)

            self.canvas.setFocus(True)
            return True
        return False   
        
    def openNextVisual(self, _value=False):
        # Proceding prev image without dialog if having any label
        
        print("its here testing Visual thing", self.filePath)
        self.loadFile(filePath=self.filePath)
        
    def QImageToCvMat(incomingImage):
        '''  Converts a QImage into an opencv MAT format  '''

        incomingImage = incomingImage.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)

        width = incomingImage.width()
        height = incomingImage.height()

        ptr = incomingImage.bits()
        ptr.setsize(height * width * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        
        return arr
        
    def refresh(self):
        elements = [c for c in self.selection_group.children() if isinstance(c, QRadioButton)]
        for idx, radio in enumerate(elements):
            radio.blockSignals(True)
            if not idx:
                radio.setChecked(True)
            radio.blockSignals(False)

        elements = [c for c in self.tags_group.children() if isinstance(c, QCheckBox)]
        for check in elements:
            check.blockSignals(True)
            check.setChecked(False)
            check.blockSignals(False)

    def fetchDatabase(self):

        currIndex = self.mImgList.index(self.filePath)
        filename = self.mImgList[currIndex]

        self.refresh()

        

        if os.path.basename(self.filePath) in self.database.load():
            data = self.database.load()[os.path.basename(self.filePath)]
            # print(f'data: {data}')

            if data is None:
                print(f'{filename} is not Annotated yet!')
                return

            elements = [c for c in self.selection_group.children() if isinstance(c, QRadioButton)]
            # print(f'Elements: {elements}')
            for radio in elements:
                radio.blockSignals(True)
            for radio in elements:
                # print(f'Radio: {radio.text()}')
                if radio.text().split(': ')[1] == data['selection']:
                    radio.setChecked(True)
            for radio in elements:
                radio.blockSignals(False)

            # Check tags
            elements = [c for c in self.tags_group.children() if isinstance(c, QCheckBox)]
            for check in elements:
                if check.text().split(': ')[1] in data['tags']:
                    check.setChecked(True)
        else:
            pass

    def updateDatabase(self):
        # Check selection type
        currIndex = self.mImgList.index(self.filePath)
        filename = self.mImgList[currIndex]
        # print(f'Update Database Filename: {filename}')
        elements = [c for c in self.selection_group.children() if isinstance(c, QRadioButton)]
        selected = ''
        for radio in elements:
            if radio.isChecked():
                selected = radio.text().split(': ')[1]

        # Check tags
        elements = [c for c in self.tags_group.children() if isinstance(c, QCheckBox)]
        tags = []
        for check in elements:
            if check.isChecked():
                tags.append(check.text().split(': ')[1])

        msg = self.database.add_record(filename, selected, tags)
        self.statusBar().showMessage(msg)

    def save_database(self):
        self.database.save()
        self.statusBar().showMessage('Saved Successfully')

        ################################### NEW STUFF ###################################

    pyqtSlot()

    def is_toggled(self):
        btn = self.sender()
        if isinstance(btn, QRadioButton) and btn.isChecked():
            if not self.mImgList:
                pass
            elif self.mImgList is not None:
                self.updateDatabase()
        # pass

    def create_selection_dock(self):
        self.min_dock_width = 80

        selection_dock = QDockWidget('Selection', self)
        selection_dock.setObjectName(str('Selection'))

        self.selection_group = QGroupBox("")
        radio1 = QRadioButton("&P: Keep")
        radio2 = QRadioButton("&O: Annotate")
        radio3 = QRadioButton("&I: Delete")

        radio1.toggled.connect(self.is_toggled)
        radio2.toggled.connect(self.is_toggled)
        radio3.toggled.connect(self.is_toggled)
        radio1.setChecked(True)

        vbox = QVBoxLayout()
        vbox.addWidget(radio1)
        vbox.addWidget(radio2)
        vbox.addWidget(radio3)
        vbox.addStretch(1)
        self.selection_group.setLayout(vbox)

        selection_dock.setWidget(self.selection_group)
        # selection_dock.setFloating(False)
        # selection_dock.widget().setMinimumSize(QSize(self.min_dock_width, 100))
        self.addDockWidget(Qt.RightDockWidgetArea, selection_dock)

        """
        To hide the dock use:
            selection_dock.hide()
        This can be used under following circumstances:
        1. If the new data is being annotated then this can be hidden.
        2. If the annotated images are being shown then this functionality should be visible.
        3. Set a flag or an argument to the App like (Annotate or Select) to enable/disable
           this functionality
        """

    def create_tags_dock(self, tags):
        tags_dock = QDockWidget('Tags', self)
        tags_dock.setObjectName(str('Tags'))
        self.tags_group = QGroupBox("")
        self.tags_group.setFlat(True)

        alpha_indices = ('T', 'Y', 'U', 'I', 'O', 'P')
        checkboxes = []
        for tag_id, tag in tags.items():
            if tag_id < 10:
                checkboxes.append(QCheckBox(f"&{tag_id}: {tag}"))
            elif 10 <= tag_id < 16:
                checkboxes.append(QCheckBox(f"(&{alpha_indices[tag_id - 10]}){tag_id}: {tag}"))

        vbox = QVBoxLayout()

        for chk in checkboxes:
            chk.toggled.connect(self.updateDatabase)
            vbox.addWidget(chk)
        vbox.addStretch(1)
        self.tags_group.setLayout(vbox)

        tags_dock.setWidget(self.tags_group)
        # tags_dock.setFloating(False)
        tags_dock.widget().setMinimumSize(QSize(self.min_dock_width, 100))
        self.addDockWidget(Qt.RightDockWidgetArea, tags_dock)

    def validate_input(self, rgx, ceil_or_floor):
        ceil_input_validator = QRegExpValidator(rgx, ceil_or_floor)
        return ceil_or_floor.setValidator(ceil_input_validator)

    def return_loadFilename(self):
        
        currIndex = self.mImgList.index(self.filePath)
        filename = self.mImgList[currIndex]
        return self.loadFile(filename)

    @pyqtSlot()
    def test_stuff_callback(self):
        """
        :returns : Loading the file from Memory if no new rect added. If added, saving and loading the saved file.
        """

        if not self.shapes_when_file_loaded and not self.canvas.final_shapes:
            self.return_loadFilename()

        elif self.shapes_when_file_loaded and not self.canvas.final_shapes:
            self.canvas.final_shapes.append(self.shapes_when_file_loaded)
            self.shapeCountBefore = len(self.shapes_when_file_loaded)
            self.return_loadFilename()

        elif self.shapeCountBefore == len(self.canvas.shapes):
            self.return_loadFilename()
            print(f'canvas shapes: {self.canvas.shapes}')

        elif self.shapeCountBefore < len(self.canvas.shapes):
            self.saveFile(True)
            self.return_loadFilename()
            self.shapeCountBefore = len(self.canvas.shapes)

        # print(f'callBack: {self.shapeCountBefore}')
        # elif self.shapeCountBefore > len(self.canvas.shapes):
        #     self.saveFile(True)
        #     self.return_loadFilename()
        #     print('rect was removed')
        #     print(self.shapeCountBefore)

    def test_stuff_here(self):
        new_dock = QDockWidget("Test", self)

        # Validate Input Regular Expression
        rgx = QRegExp("^[0-9]$|^[1-9][0-9]$|^(100)$")

        self.ceil_label = QLabel()
        self.ceil_label.setText('Ceil: ')

        self.floor_label = QLabel()
        self.floor_label.setText('Floor: ')

        self.ceil_le = QLineEdit()
        self.ceil_le.setText(str(90))
        self.validate_input(rgx, self.ceil_le)
        self.ceil_le.setObjectName("Ceil")
        # print(f'Ceil Text: {int(self.ceil_le.text())}')

        self.floor_le = QLineEdit()
        self.floor_le.setText(str(10))
        self.validate_input(rgx, self.floor_le)
        self.floor_le.setObjectName("Floor")
        # print(f'Floor text: {int(self.floor_le.text())}')

        self.pb = QPushButton("Change")
        self.pb.setObjectName("connect")

        if self.saveFile:
            self.pb.clicked.connect(self.test_stuff_callback)

        hbox = QHBoxLayout()
        # hbox.addStretch(1)
        hbox.addWidget(self.ceil_label)
        hbox.addWidget(self.ceil_le)
        hbox.addWidget(self.floor_label)
        hbox.addWidget(self.floor_le)
        hbox.addWidget(self.pb)

        vbox = QHBoxLayout()
        vbox.addStretch(1)
        vbox.addLayout(hbox)

        buttons_widget = QWidget()
        buttons_widget.setLayout(vbox)
        new_dock.setWidget(buttons_widget)
        new_dock.setFloating(False)
        new_dock.widget().setMinimumSize(QSize(self.min_dock_width, 80))
        self.addDockWidget(Qt.RightDockWidgetArea, new_dock)
        
    
        
        ################################### NEW STUFF ###################################
    
    
    
    
    def openFile(self, _value=False):
        if not self.mayContinue():
            return
        path = os.path.dirname(ustr(self.filePath)) if self.filePath else '.'
        formats = ['*.%s' % fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
        filters = "Image & Label files (%s)" % ' '.join(formats + ['*%s' % LabelFile.suffix])
        filename = QFileDialog.getOpenFileName(self, '%s - Choose Image or Label file' % __appname__, path, filters)
        if filename:
            if isinstance(filename, (tuple, list)):
                filename = filename[0]
            self.loadFile(filename)

    def saveFile(self, _value=False):
        print('Inside saveFile function')
        # Get the sesison name by parsing the string
        # prefix = 'xl_visual_session' + '_' + '_'.join(self.session_name.split('_')[:2]) + '_'  # xl_visual_session_xlg31025_20200902_
        prefix = 'xl_'
        if self.defaultSaveDir is not None and len(ustr(self.defaultSaveDir)):
            print('here')
            if self.filePath:
                print('here!!!!')
               
                imgFileName = os.path.basename(self.filePath)
                print(f'imgFileName: {imgFileName}')
                savedFileName = os.path.splitext(imgFileName)[0]
                print(f'savedFileName: {savedFileName}')
                split_filename = savedFileName.split('_')
                print(f'split_filename: {split_filename}')
                newFileName = split_filename[0] + '_' + split_filename[1] #+ '_' + 'od' + '_' + split_filename[2]
                print(f'newFileName: {newFileName}')
                comp_filename = prefix + split_filename[-1]  ## New stuff
                print(f'comp filename: {comp_filename}')
                savedPath = os.path.join(ustr(self.defaultSaveDir), comp_filename)  # change savedFileName to newFileName
                print(f'savedPath for Annotation File: {savedPath}')
                self._saveFile(savedPath)
                '''
                imgFileName = os.path.basename(self.filePath)
                savedPath= " G:/One/Annotation_OD_refined/" + imgFileName.replace(".vis","")
                print("_______", savedPath)
                self._saveFile(savedPath)
                '''
                ###################### NEW STUFF ##################
                self.save_database()
                self.return_loadFilename()
                ###################### NEW STUFF ##################
        else:
            print('else here')
            imgFileDir = os.path.dirname(self.filePath)
            imgFileName = os.path.basename(self.filePath)
            savedFileName = os.path.splitext(imgFileName)[0]
            savedPath = os.path.join(imgFileDir, savedFileName)
            self._saveFile(savedPath if self.labelFile
                           else self.saveFileDialog(removeExt=False))

            ###################### NEW STUFF ##################
            self.save_database()
            ###################### NEW STUFF ##################

    def saveFileAs(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        self._saveFile(self.saveFileDialog())

    def saveFileDialog(self, removeExt=True):
        caption = '%s - Choose File' % __appname__
        filters = 'File (*%s)' % LabelFile.suffix
        openDialogPath = self.currentPath()
        dlg = QFileDialog(self, caption, openDialogPath, filters)
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QFileDialog.AcceptSave)
        filenameWithoutExtension = os.path.splitext(self.filePath)[0]
        dlg.selectFile(filenameWithoutExtension)
        dlg.setOption(QFileDialog.DontUseNativeDialog, False)
        if dlg.exec_():
            fullFilePath = ustr(dlg.selectedFiles()[0])
            if removeExt:
                return os.path.splitext(fullFilePath)[0]  # Return file path without the extension.
            else:
                return fullFilePath
        return ''

    def _saveFile(self, annotationFilePath):
        if annotationFilePath and self.saveLabels(annotationFilePath):
            print(f'saveLabel: {self.saveLabels(annotationFilePath)}')
            self.setClean()
            self.statusBar().showMessage('Saved to  %s' % annotationFilePath)
            self.statusBar().show()

    def closeFile(self, _value=False):
        if not self.mayContinue():
            return
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.actions.saveAs.setEnabled(False)

    def deleteImg(self):
        deletePath = self.filePath
        if deletePath is not None:
            self.openNextImg()
            if os.path.exists(deletePath):
                os.remove(deletePath)
            self.importDirImages(self.lastOpenDir)

    def resetAll(self):
        self.settings.reset()
        self.close()
        proc = QProcess()
        proc.startDetached(os.path.abspath(__file__))

    def mayContinue(self):
        if not self.dirty:
            return True
        else:
            discardChanges = self.discardChangesDialog()
            if discardChanges == QMessageBox.No:
                return True
            elif discardChanges == QMessageBox.Yes:
                self.saveFile()
                return True
            else:
                return False

    def discardChangesDialog(self):
        yes, no, cancel = QMessageBox.Yes, QMessageBox.No, QMessageBox.Cancel
        msg = u'You have unsaved changes, would you like to save them and proceed?\nClick "No" to undo all changes.'
        return QMessageBox.warning(self, u'Attention', msg, yes | no | cancel)

    def errorMessage(self, title, message):
        return QMessageBox.critical(self, title,
                                    '<p><b>%s</b></p>%s' % (title, message))

    def currentPath(self):
        return os.path.dirname(self.filePath) if self.filePath else '.'

    def chooseColor1(self):
        color = self.colorDialog.getColor(self.lineColor, u'Choose line color',
                                          default=DEFAULT_LINE_COLOR)
        if color:
            self.lineColor = color
            Shape.line_color = color
            self.canvas.setDrawingColor(color)
            self.canvas.update()
            self.setDirty()

    def deleteSelectedShape(self):
        self.remLabel(self.canvas.deleteSelected())
        self.setDirty()
        if self.noShapes():
            for action in self.actions.onShapesPresent:
                action.setEnabled(False)
            print('no')
        self.saveFile(True)
        self.shapeCountBefore = len(self.canvas.shapes)

    def chshapeLineColor(self):
        color = self.colorDialog.getColor(self.lineColor, u'Choose line color',
                                          default=DEFAULT_LINE_COLOR)
        if color:
            self.canvas.selectedShape.line_color = color
            self.canvas.update()
            self.setDirty()

    def chshapeFillColor(self):
        color = self.colorDialog.getColor(self.fillColor, u'Choose fill color',
                                          default=DEFAULT_FILL_COLOR)
        if color:
            self.canvas.selectedShape.fill_color = color
            self.canvas.update()
            self.setDirty()

    def copyShape(self):
        self.canvas.endMove(copy=True)
        self.addLabel(self.canvas.selectedShape)
        self.setDirty()

    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()

    def loadPredefinedClasses(self, predefClassesFile):
        if os.path.exists(predefClassesFile) is True:
            with codecs.open(predefClassesFile, 'r', 'utf8') as f:
                for line in f:
                    line = line.strip()
                    if self.labelHist is None:
                        self.labelHist = [line]
                    else:
                        self.labelHist.append(line)

    def loadPascalXMLByFilename(self, xmlPath):
        print(self.filePath)
        if self.filePath is None:
            return
        if os.path.isfile(xmlPath) is False:
            return

        self.set_format(FORMAT_PASCALVOC)

        tVocParseReader = PascalVocReader(xmlPath)
        shapes = tVocParseReader.getShapes()
        # print(shapes)
        self.loadLabels(shapes)
        self.canvas.verified = tVocParseReader.verified

    def loadYOLOTXTByFilename(self, txtPath):
        if self.filePath is None:
            return
        if os.path.isfile(txtPath) is False:
            return

        # self.set_format(FORMAT_YOLO)
        tYoloParseReader = YoloReader(txtPath, self.image)
        shapes = tYoloParseReader.getShapes()
        # print(shapes)
        self.loadLabels(shapes)
        self.canvas.verified = tYoloParseReader.verified

    
    
    
    def read_pointCloud_distance(self,filepath):
        pnt_ = self.read_matrix(filepath)
        points=[]
        #5137*13
        for i in range(pnt_.shape[0]):
             
            x = pnt_[:,4:6][i][0]
            y = pnt_[:,4:6][i][1]
            d = pnt_[i,6]

            if x<0 or y<0:
                pass
            else:
                points.append((x,y,d))
                
        return points

    def loadXLTXTByFilename(self, txtPath, Transform=False):
        if self.filePath is None:
            return
        # print(f'In loadXLTxT fileName: {os.path.isfile(txtPath)}')
        if os.path.isfile(txtPath) is False:
            return

        self.set_format(FORMAT_XL)
        tXLParseReader = XLReader(txtPath, self.image)
        # print(f'shapes from the file: {tXLParseReader.getShapes()}')
        shapes = tXLParseReader.getShapes()
        
        
            
        point_cloud_file = self.filePath.replace(".vis",".xpc").replace("xl_visual", "xw_pointcloud")
        points = self.read_pointCloud_distance(point_cloud_file)
        
        
      
        box_points = [[] for x in range(len(shapes))]
        for point in points:
            #print((point[0], point[1]))
            for i, boxes in enumerate(shapes):
                
                xmin = boxes[1][0][0]
                xmax = boxes[1][2][0]
                ymin = boxes[1][0][1]
                ymax = boxes[1][2][1]
                
                #print("min", (xmin,ymin))
                #print("min", (xmin,ymin))
                if point[0] < xmin or point[0]>xmax:
                    continue
                elif point[1] < ymin or point[1]>ymax:
                    continue
                else:
                    box_points[i].append(point[2])
        #print(box_points)
        
        distances = []
        for i in box_points:
            if len(i)==0:
                distances.append(0)
            else:
                distances.append(self.median(i))
            #self.shape.test_distance = self.median(i)
        #print(np.asarray(distances)/1000)
        
        self.distances = distances
        #self.shape.test_array = distances
        #print(self.distances)
        self.loadLabels(shapes, Transform, distances)
        
        self.canvas.verified = tXLParseReader.verified
    
    def median(self,lst):
        lst.sort()  # Sort the list first
        if len(lst) % 2 == 0:  # Checking if the length is even
            # Applying formula which is sum of middle two divided by 2
            
            return (lst[len(lst) // 2] + lst[(len(lst) - 1) // 2]) / 2
        else:
            
            # If length is odd then get middle value
            return lst[len(lst) // 2]

    ################################################## NEW STUFF ##############################################
    def loadCreateMLJSONByFilename(self, jsonPath, filePath):
        if self.filePath is None:
            return
        if os.path.isfile(jsonPath) is False:
            return

        self.set_format(FORMAT_CREATEML)

        crmlParseReader = CreateMLReader(jsonPath, filePath)
        shapes = crmlParseReader.get_shapes()
        self.loadLabels(shapes)
        self.canvas.verified = crmlParseReader.verified

    def copyPreviousBoundingBoxes(self):
        currIndex = self.mImgList.index(self.filePath)
        if currIndex - 1 >= 0:
            prevFilePath = self.mImgList[currIndex - 1]
            self.showBoundingBoxFromAnnotationFile(prevFilePath)
            self.saveFile()

    def togglePaintLabelsOption(self):
        for shape in self.canvas.shapes:
            shape.paintLabel = self.displayLabelOption.isChecked()
        

    def toogleDrawSquare(self):
        self.canvas.setDrawingShapeToSquare(self.drawSquaresOption.isChecked())


def inverted(color):
    return QColor(*[255 - v for v in color.getRgb()])


def read(filename, default=None):
    try:
        reader = QImageReader(filename)
        reader.setAutoTransform(True)
        return reader.read()
    except:
        return default


def get_main_app(argv=[]):
    """
    Standard boilerplate Qt application code.
    Do everything but app.exec_() -- so that we can test the application in one thread
    """
    app = QApplication(argv)
    app.setApplicationName(__appname__)
    app.setWindowIcon(newIcon("app"))
    # Tzutalin 201705+: Accept extra agruments to change predefined class file
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--image_dir", nargs="?")
    argparser.add_argument("--predefined_classes_file",
                           default=os.path.join(os.path.dirname(__file__), "data", "predefined_classes.txt"),
                           nargs="?")
    argparser.add_argument("--save_dir", nargs="?")
    argparser.add_argument("--which_type", type=str, nargs="?")
    args = argparser.parse_args(argv[1:])
    # Usage : labelImg.py image predefClassFile saveDir

    ########################################### NEW STUFF ###############################
    script_path = Path(__file__).resolve().parent
    with script_path.joinpath('config.yaml').open('r') as yf:
        config = yaml.safe_load(yf)
    win = MainWindow(config['tags'],
                     args.image_dir,
                     args.predefined_classes_file,
                     args.save_dir)
    # args.format)
    ########################################### NEW STUFF ###############################
    win.show()
    return app, win


def main():
    '''construct main app and run it'''
    app, _win = get_main_app(sys.argv)
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())

# Regex: ^(?:[1-9][0-9]{0,4}(?:\.\d{1,2})?|65536|)$
