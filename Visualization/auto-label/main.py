import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QLabel,
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


class SegmentationEditor(QWidget):
    def __init__(self, image_path, mask_path):
        super().__init__()
        self.setWindowTitle("Segmentation Editor")
        self.image_path = image_path
        self.mask_path = mask_path
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Load image and mask
        self.image = cv2.imread(self.image_path, cv2.COLOR_BGR2RGB)
        # self.image = cv2.cvtColor(self.image)
        self.mask = cv2.imread(self.mask_path, cv2.COLOR_BGR2RGB)

        # Create scene and view
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)

        # Combine image and mask for overlay
        self.combined_image = self.overlay_mask(self.image, self.mask)
        self.pixmap_item = QGraphicsPixmapItem(self.numpy_to_pixmap(self.combined_image))
        self.scene.addItem(self.pixmap_item)

        # Buttons
        self.save_button = QPushButton("Save Changes")
        self.save_button.clicked.connect(self.save_changes)

        self.info_label = QLabel("Drag to modify the mask.")

        layout.addWidget(self.view)
        layout.addWidget(self.info_label)
        layout.addWidget(self.save_button)
        self.setLayout(layout)

        # Mouse state
        self.drawing = False

        # Enable mouse events
        self.view.viewport().installEventFilter(self)

    def overlay_mask(self, image, mask):
        """Semantic segmentation mask를 시각적으로 오버레이."""
        overlay = image.copy()
        
        # # 클래스 ID에 대한 색상 매핑 (예: 클래스 0, 1, 2 등)
        # class_colors = {
        #     (0, 0, 0): [0, 0, 0],       # 배경 (검정색)
        #     (236, 34, 237): [236, 34, 237],     # 클래스 1 (빨강)
        #     (74, 158, 201): [74, 158, 201],     # 클래스 2 (초록)
        #     (192, 32,96): [192, 32,96 ],     # 클래스 3 (파랑)
        #     (179, 134,89): [179, 134,89 ],   # 클래스 4 (노랑)
        #     (219, 223,153): [219, 223,153 ],   # 클래스 5 (자주)
        #     (77, 106,255): [77, 106,255 ],   # 클래스 6 (하늘색)
        #     (252, 100,22): [252, 100,22 ], # 클래스 7 (흰색)
        #     (45, 182,143): [45, 182,143 ],     # 클래스 8 (짙은 빨강)
        #     (129, 198,38): [129, 198,38 ],     # 클래스 9 (짙은 초록)
        #     (218, 154,27): [218, 154,27 ],    # 클래스 10 (짙은 파랑)
            
        # }

        # # 컬러 마스크를 클래스 ID로 변환
        # class_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
        # for color, class_id in enumerate(class_colors.keys()):
        #     matches = np.all(mask == np.array(color, dtype=np.uint8), axis=-1)
        #     class_mask[matches] = class_id

        # # 마스크와 이미지 오버레이
        # for class_id, color in enumerate(class_colors.values()):
        #     overlay[class_mask == class_id] = color

        # 오버레이 투명도 적용
        alpha = 0.5
        overlay = cv2.addWeighted(overlay, alpha, mask, 1 - alpha, 0)
        return overlay

    def numpy_to_pixmap(self, array):
        """Convert a NumPy array to QPixmap."""
        height, width, channel = array.shape
        bytes_per_line = 3 * width
        qimage = QImage(array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimage)

    def eventFilter(self, source, event):
        """Handle mouse events for drawing."""
        if source == self.view.viewport():
            if event.type() == event.MouseButtonPress and event.button() == Qt.LeftButton:
                self.drawing = True
                self.modify_mask(event)
            elif event.type() == event.MouseButtonRelease and event.button() == Qt.LeftButton:
                self.drawing = False
            elif event.type() == event.MouseMove and self.drawing:
                self.modify_mask(event)
        return super().eventFilter(source, event)

    def modify_mask(self, event):
        """Modify the mask based on mouse position."""
        pos = event.pos()
        scene_pos = self.view.mapToScene(pos)
        x, y = int(scene_pos.x()), int(scene_pos.y())

        if 0 <= x < self.mask.shape[1] and 0 <= y < self.mask.shape[0]:
            cv2.circle(self.mask, (x, y), 5, 255, -1)  # Draw on the mask
            self.combined_image = self.overlay_mask(self.image, self.mask)
            self.pixmap_item.setPixmap(self.numpy_to_pixmap(self.combined_image))

    def save_changes(self):
        """Save the modified mask."""
        cv2.imwrite("modified_mask.png", self.mask)
        self.info_label.setText("Changes saved as 'modified_mask.png'.")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Paths to image and mask
    image_path = "E:/dataset/foot_ball_segmenatation/images/1.png"  # Replace with your image path
    mask_path = "E:/dataset/foot_ball_segmenatation/masks/1.png"  # Replace with your mask path

    editor = SegmentationEditor(image_path, mask_path)
    editor.resize(800, 600)
    editor.show()

    sys.exit(app.exec_())
