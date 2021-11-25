import os
from pathlib import Path
from qtpy.QtWidgets import QListWidget
from qtpy.QtCore import Qt


class FolderListWidget(QListWidget):
    # be able to pass the Napari viewer name (viewer)
    def __init__(self, viewer, parent=None):
        super().__init__(parent)

        self.viewer = viewer
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.currentItemChanged.connect(self.open_file)

        self.folder_path = None

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):

        self.clear()
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.CopyAction)
            event.accept()
            # Check that it's a LIF file
            for url in event.mimeData().urls():
                self.folder_path = str(url.toLocalFile())
                files = os.listdir(self.folder_path)  
                for f in files:
                    #if Path(f).suffix == '.oir':
                    if f[0] != '.':
                        self.addItem(f)

    def update_from_path(self, path):

        self.clear()
        self.folder_path = path
        files = os.listdir(self.folder_path)  
        for f in files:
            #if Path(f).suffix == '.oir':
            if f[0] != '.':
                self.addItem(f)

    def select_first_file(self):
        
        self.setCurrentRow(0)

    def open_file(self):
        item = self.currentItem()
        image_name = item.text()
        self.viewer.layers.clear()
        self.viewer.open(Path(self.folder_path).joinpath(image_name))