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
from app.ui.io_page import IoPage
from app.ui.config_page import ConfigPage 
from app.config_manager import ConfigManager

# --- คลาสหน้าต่างหลัก ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Vision Ubend")
        self.setWindowIcon(QIcon("icon.png"))
        self.config = ConfigManager()
        # --- ส่วนจัดการรหัสผ่าน ---
        self.PASSWORD = "1234"
        self.is_unlocked = False
        self.is_checking_password = False
        self.locked_tabs_indices = {1, 2, 3}

        self.unlock_timestamp = 0
        self.LOCK_DURATION = 10

        # --- สร้างและตั้งค่า QTabWidget ---
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.West)
        self.tabs.tabBar().setExpanding(True) # คำสั่งนี้ถูกต้องแล้ว

        # --- เพิ่มหน้าต่างๆ โดยใช้คลาสที่ Import มา ---
        self.tabs.addTab(HomePage(self.config), "HOME")
        self.tabs.addTab(SettingPage(), "SETTINGS")
        self.tabs.addTab(IoPage(), "I/O")
        self.tabs.addTab(ConfigPage(self.config), "CONNECTION")
        self.tabs.addTab(QWidget(), "EXIT")

        # --- ตั้งค่า Stylesheet (ฉบับแก้ไขเพื่อให้แท็บขยายได้) ---
        self.tabs.setStyleSheet("""
            /* พื้นหลังส่วนเนื้อหาของ Tab */
            QTabWidget::pane {
                border: none;
                background-color: #FFFFFF;
            }

            /* แถบ Tab ทั้งหมด (พื้นที่ด้านข้าง) */
            QTabBar {
                background-color: #FFFFFF; /* 🎯 ทำให้พื้นหลังเป็นสีขาว */
                min-width: 50px;
                min-height: 50px;
                border: none;
                /* เพิ่มเส้นคั่นด้านขวาของแถบทั้งหมด */
                border-right: 2px solid #F0F0F0;
            }

            /* ปุ่มแท็บแต่ละอัน */
            QTabBar::tab {
                background-color: transparent; /* ทำให้พื้นหลังโปร่งใสในสถานะปกติ */
                color: #333333;
                padding: 56px 14px; /* 🎯 เพิ่ม padding แนวตั้งเพื่อให้ดู 'ยาวขึ้น' */
                font-size: 16px;
                font-weight: bold;
                border: none; /* เอาเส้นขอบรอบปุ่มออก */
            }
            
            /* ปุ่มแท็บที่กำลังถูกเลือก */
            QTabBar::tab:selected {
                background-color: #003366;
                color: #FFFFFF;
            }

            /* ปุ่มแท็บเมื่อเมาส์ชี้อยู่ */
            QTabBar::tab:hover {
                background-color: #E0F2FF;
            }
             /* 🎯 เพิ่มสไตล์พิเศษสำหรับปุ่ม EXIT (แท็บสุดท้าย) */
            QTabBar::tab:last {
                background-color: #D32F2F; /* สีแดงเข้ม */
                color: white;
            }
            QTabBar::tab:last:hover {
                background-color: #F44336; /* สีแดงสว่างขึ้นเมื่อเมาส์ชี้ */
            }                    

        """)

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
