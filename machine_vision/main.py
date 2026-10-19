import sys
import time
from PySide6.QtWidgets import ( # type: ignore
    QApplication, QMainWindow, QTabWidget, QWidget,
    QLabel, QVBoxLayout, QInputDialog, QLineEdit, QMessageBox
)
from PySide6.QtGui import QIcon, QFont # type: ignore
from PySide6.QtCore import Qt # type: ignore

# --- Import คลาสจากไฟล์ที่แยกไว้ ---
# ตรวจสอบให้แน่ใจว่าคุณมีไฟล์เหล่านี้อยู่ในโฟลเดอร์เดียวกัน
from app.ui.home_page import HomePage
from app.ui.setting_page import SettingPage
from app.ui.config_page import ConfigPage 
from app.ui.report_page import ReportPage
from app.config_manager import ConfigManager

# --- คลาสหน้าต่างหลัก ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Vision Ubend")
        self.setWindowIcon(QIcon("icon.png"))
        self.config = ConfigManager()

        # password/lock settings
        self.PASSWORD = "1234"
        self.is_unlocked = False
        self.is_checking_password = False
        # lock SETTINGS (2), CONNECTION (3), EXIT (4). Report (index 1) is unlocked
        self.locked_tabs_indices = {2, 3, 4}
        self.unlock_timestamp = 0
        self.LOCK_DURATION = 10

        # tabs
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.West)
        self.tabs.tabBar().setExpanding(True)

        # create pages
        self.home_page = HomePage(self.config)
        self.report_page = ReportPage()
        self.setting_page = SettingPage()
        self.config_page = ConfigPage(self.config)

        # connect signals (Home -> Setting)
        self.home_page.oring_status.connect(self.setting_page.update_oring)
        self.home_page.emit_summary.connect(self.setting_page.parse_and_update)
        # OK/NG settings: SETTINGS -> HOME (used from the next frame), HOME -> SETTINGS (live preview / measured values)
        self.setting_page.oring_settings_changed.connect(self.home_page.apply_oring_settings)
        self.home_page.parts_evaluated.connect(self.setting_page.on_parts_evaluated)

        # add tabs
        self.tabs.addTab(self.home_page, "HOME")           # index 0
        self.tabs.addTab(self.report_page, "REPORT")       # index 1 (unlocked)
        self.tabs.addTab(self.setting_page, "SETTINGS")    # index 2 (locked)
        self.tabs.addTab(self.config_page, "CONNECTION")   # index 3 (locked)
        self.tabs.addTab(QWidget(), "EXIT")                # index 4 (locked)

        # stylesheet
        self.tabs.setStyleSheet(
            """
            QTabWidget::pane { border: none; background-color: #FFFFFF; }
            QTabBar { background-color: #FFFFFF; min-width: 50px; min-height: 50px; border: none; border-right: 2px solid #F0F0F0; }
            QTabBar::tab { background-color: transparent; color: #333333; padding: 73px 14px; font-size: 16px; font-weight: bold; border: none; }
            QTabBar::tab:selected { background-color: #003366; color: #FFFFFF; }
            QTabBar::tab:hover { background-color: #E0F2FF; }
            QTabBar::tab:last { background-color: #D32F2F; color: white; }
            QTabBar::tab:last:hover { background-color: #F44336; }
            """
        )

        self.tabs.currentChanged.connect(self.on_tab_changed)
        self.setCentralWidget(self.tabs)

    def on_tab_changed(self, index):
        if self.tabs.tabText(index) == "EXIT":
            # --- สร้าง Popup ถามยืนยัน ---
            reply = QMessageBox.question(self, 
                                         'Confirm Exit',
                                         'คุณต้องการออกจากโปรแกรมใช่หรือไม่?',
                                         QMessageBox.Yes | QMessageBox.No, 
                                         QMessageBox.No)

            if reply == QMessageBox.Yes:
                self.close() # ถ้าใช่, ปิดโปรแกรม
            else:
                # ถ้าไม่, บังคับให้กลับไปที่ Tab แรก
                self.tabs.blockSignals(True)
                self.tabs.setCurrentIndex(0)
                self.tabs.blockSignals(False)
            return 
        
        if self.is_checking_password:
            return

        if self.is_unlocked and (time.time() - self.unlock_timestamp > self.LOCK_DURATION):
            # print(f"Session หมดอายุหลังจาก {self.LOCK_DURATION / 60} นาที, กรุณาใส่รหัสผ่านอีกครั้ง")
            self.is_unlocked = False

        if index in self.locked_tabs_indices and not self.is_unlocked:
            self.is_checking_password = True
            password, ok = QInputDialog.getText(
                self, "Password Required", "Enter password to access:", QLineEdit.Password
            )
            if ok and password == self.PASSWORD:
                self.is_unlocked = True
                self.unlock_timestamp = time.time()
                # print(f"ปลดล็อกสำเร็จ! ระบบจะล็อกอีกครั้งใน {self.LOCK_DURATION / 60} นาที")
            else:
                # if ok:
                #     return
                    # print("รหัสผ่านไม่ถูกต้อง")
                self.tabs.blockSignals(True)
                self.tabs.setCurrentIndex(0)
                self.tabs.blockSignals(False)
            self.is_checking_password = False

    def closeEvent(self, event):
        # สั่งให้ Worker ของหน้า Home หยุดทำงานก่อนปิดโปรแกรม
        home_widget = self.tabs.widget(0) # Index 0 คือ HomePage
        if hasattr(home_widget, 'camera_worker') and home_widget.camera_worker:
            home_widget.camera_worker.stop()
        event.accept()    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    
    # --- ตัวเลือกในการปรับขนาดหน้าต่าง ---
    # window.showMaximized() 
    # 1. ตั้งค่าขนาดเริ่มต้นที่กำหนดเองและอนุญาตให้ปรับขนาดได้ (แนะนำหากต้องการปรับขนาด)
    # window.resize(1280, 720) # ตัวอย่าง: ตั้งค่าความกว้าง 1280 พิกเซล, ความสูง 720 พิกเซล
    # window.show()

    # 2. ตั้งค่าขนาดคงที่ (ไม่สามารถปรับขนาดได้)
    # หากต้องการขนาดที่แน่นอนและไม่ให้ผู้ใช้ปรับได้
    # window.setFixedSize(800, 600) # ตัวอย่าง: ตั้งค่าเป็น 800x600 พิกเซล และล็อกขนาด
    # window.show()

    # 3. เปิดหน้าต่างแบบเต็มจอ (แต่ยังเห็น Taskbar)
    # window.showMaximized()

    # 4. โค้ดเดิม: เปิดแบบเต็มหน้าจอ (ซ่อน Taskbar) และล็อกขนาด
    screen_geometry = QApplication.primaryScreen().geometry()
    window.setFixedSize(screen_geometry.width(), screen_geometry.height())
    window.showFullScreen()
    
    sys.exit(app.exec())
