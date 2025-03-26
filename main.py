import sys
import cv2
import warnings
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from win_ui.win_ui import MainWinUI  # 引入UI窗口
warnings.filterwarnings("ignore")


class VideoDetectionThread(QThread):
    # 信号用于传递原始图像、检测结果图像和检测信息
    frame_processed = pyqtSignal(QImage, QImage, str)

    def __init__(self, detector, video_capture, parent=None):
        super().__init__(parent)
        self.detector = detector
        self.video_capture = video_capture
        self._run_flag = True

    def run(self):
        while self._run_flag:
            ret, frame = self.video_capture.read()
            if not ret:
                break
            # 将BGR转换为RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            bytes_per_line = 3 * w
            # 创建原始图像的QImage
            original_qimage = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            # 推理处理
            blob, original_shape, scale = self.detector.preprocess(frame_rgb)
            self.detector.net.setInput(blob)
            outputs = self.detector.net.forward()
            boxes, scores, class_ids = self.detector.postprocess(outputs, original_shape, scale)
            detected_frame = frame_rgb.copy()
            detected_frame, class_counts = self.detector.draw_detections(detected_frame, boxes, scores, class_ids)
            # 创建检测结果图像的QImage
            detected_qimage = QImage(detected_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            # 生成检测信息文本
            info_text = (f"当前模型: {self.detector.current_model}\n"
                         f"置信度: {self.detector.confidence_threshold:.2f}\n"
                         f"IOU阈值: {self.detector.iou_threshold:.2f}\n"
                         f"检测结果：\n")
            for name, count in class_counts.items():
                info_text += f"  {name}: {count}\n"
            # 发送信号到主线程更新UI
            self.frame_processed.emit(original_qimage, detected_qimage, info_text)
            # 控制下帧率（这里睡眠30ms，相当于大约33FPS的上限）
            self.msleep(30)

    def stop(self):
        self._run_flag = False
        self.wait()


class YOLOv8ONNXDetector(QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = MainWinUI()  # 创建UI实例
        self.ui.setupUi(self)  # 初始化UI

        # 模型相关参数
        self.net = None
        self.input_size = 640
        self.class_names = self.load_class_names("./pt/coco.names")
        self.current_model = ""

        # 检测参数
        self.confidence_threshold = 0.25
        self.iou_threshold = 0.45
        self.score_threshold = 0.2

        # 视频线程对象
        self.video_thread = None
        self.video_capture = None

        # 初始化模型选择器
        self.ui.model_selector.addItem("yolo11n.onnx")
        self.ui.model_selector.addItem("yolo11s.onnx")
        self.ui.model_selector.addItem("yolo11m.onnx")
        self.ui.model_selector.addItem("yolo11l.onnx")
        self.ui.model_selector.addItem("yolo11x.onnx")
        self.load_onnx_model('./pt/yolo11n.onnx')

        # 连接信号
        self.ui.load_image_button.clicked.connect(self.load_image)
        self.ui.load_video_button.clicked.connect(self.load_video)
        self.ui.camera_button.clicked.connect(self.start_camera)
        self.ui.stop_camera_button.clicked.connect(self.stop_camera)
        self.ui.model_selector.currentIndexChanged.connect(self.model_selection_changed)
        self.ui.confidence_slider.valueChanged.connect(self.update_confidence)
        self.ui.iou_slider.valueChanged.connect(self.update_iou)

    def load_onnx_model(self, model_path):
        """加载ONNX模型"""
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.current_model = model_path.split('/')[-1]

    def model_selection_changed(self):
        selected_model = self.ui.model_selector.currentText()
        self.load_onnx_model(f'./pt/{selected_model}')

    def preprocess(self, image):
        """图像预处理：缩放、填充、转换为blob"""
        h, w = image.shape[:2]
        scale = min(self.input_size / h, self.input_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        new_image = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        new_image[:new_h, :new_w] = resized
        blob = cv2.dnn.blobFromImage(new_image, 1 / 255.0, (self.input_size, self.input_size),
                                     swapRB=True, crop=False)
        return blob, (h, w), scale

    def postprocess(self, outputs, original_shape, scale):
        """后处理：解析输出、过滤低置信度、坐标转换、NMS"""
        h, w = original_shape
        outputs = outputs[0].transpose(1, 0)
        boxes, scores, class_ids = [], [], []
        for output in outputs:
            max_score = np.max(output[4:])
            if max_score < self.score_threshold:
                continue
            cx, cy = output[0], output[1]
            bw, bh = output[2], output[3]
            x1 = int((cx - bw / 2) / scale)
            y1 = int((cy - bh / 2) / scale)
            x2 = int((cx + bw / 2) / scale)
            y2 = int((cy + bh / 2) / scale)
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            class_id = np.argmax(output[4:])
            boxes.append([x1, y1, x2, y2])
            scores.append(max_score)
            class_ids.append(class_id)
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_threshold, self.iou_threshold)
        final_boxes, final_scores, final_class_ids = [], [], []
        if len(indices) > 0:
            for i in indices.flatten():
                final_boxes.append(boxes[i])
                final_scores.append(scores[i])
                final_class_ids.append(class_ids[i])
        return final_boxes, final_scores, final_class_ids

    def draw_detections(self, image, boxes, scores, class_ids):
        """绘制检测结果：绘制边框、标签、统计检测数量"""
        class_counts = {}
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box
            class_name = self.class_names[class_id]
            color = self.get_color(class_id)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name}: {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - 20), (x1 + tw, y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        return image, class_counts

    @staticmethod
    def get_color(class_id):
        """为不同类别生成固定颜色"""
        np.random.seed(class_id)
        return int(np.random.randint(0, 255)), int(np.random.randint(0, 255)), int(np.random.randint(0, 255))

    @staticmethod
    def load_class_names(path):
        """加载类别名称"""
        with open(path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Images (*.png *.xpm *.jpg *.jpeg)")
        if file_name:
            frame = cv2.imread(file_name)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 直接调用单帧检测（不使用线程）
            blob, original_shape, scale = self.preprocess(frame)
            self.net.setInput(blob)
            outputs = self.net.forward()
            boxes, scores, class_ids = self.postprocess(outputs, original_shape, scale)
            detected_frame = frame.copy()
            detected_frame, class_counts = self.draw_detections(detected_frame, boxes, scores, class_ids)
            h, w = frame.shape[:2]
            bytes_per_line = 3 * w
            original_qimage = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            detected_qimage = QImage(detected_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.ui.original_label.setPixmap(QPixmap.fromImage(original_qimage).scaled(
                self.ui.original_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.ui.detected_label.setPixmap(QPixmap.fromImage(detected_qimage).scaled(
                self.ui.detected_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            info_text = (f"当前模型: {self.current_model}\n"
                         f"置信度: {self.confidence_threshold:.2f}\n"
                         f"IOU阈值: {self.iou_threshold:.2f}\n"
                         f"检测结果：\n")
            for name, count in class_counts.items():
                info_text += f"  {name}: {count}\n"
            self.ui.detect_info.setText(info_text)

    def load_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Video", "", "Videos (*.mp4 *.avi *.mov *.mkv)")
        if file_name:
            self.start_video_capture(file_name)

    def start_camera(self):
        self.start_video_capture(0)

    def start_video_capture(self, source):
        # 如果已有线程在运行，先停止
        self.stop_camera()
        self.video_capture = cv2.VideoCapture(source)
        self.video_thread = VideoDetectionThread(self, self.video_capture)
        self.video_thread.frame_processed.connect(self.update_frames)
        self.video_thread.start()

    def stop_camera(self):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        self.ui.original_label.clear()
        self.ui.detected_label.clear()
        self.ui.detect_info.clear()

    def update_frames(self, original_qimage, detected_qimage, info_text):
        # 更新原始图像显示
        pixmap_orig = QPixmap.fromImage(original_qimage).scaled(
            self.ui.original_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.ui.original_label.setPixmap(pixmap_orig)
        # 更新检测结果显示
        pixmap_detect = QPixmap.fromImage(detected_qimage).scaled(
            self.ui.detected_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.ui.detected_label.setPixmap(pixmap_detect)
        # 更新检测信息
        self.ui.detect_info.setText(info_text)

    def update_confidence(self):
        self.confidence_threshold = self.ui.confidence_slider.value() / 100.0
        self.ui.confidence_label.setText(f"置信度: {self.confidence_threshold:.2f}")

    def update_iou(self):
        self.iou_threshold = self.ui.iou_slider.value() / 100.0
        self.ui.iou_label.setText(f"IOU阈值: {self.iou_threshold:.2f}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    detector = YOLOv8ONNXDetector()
    detector.show()
    sys.exit(app.exec_())

