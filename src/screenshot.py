# -*-coding:utf-8-*-
import time
from datetime import datetime
import robomaster
from robomaster import robot
import cv2


if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False)

    print("--- คำแนะนำ ---")
    print("กด 's' บนหน้าต่างวิดีโอเพื่อบันทึกรูปภาพ (Capture)")
    print("กด 'q' เพื่อออกจากโปรแกรม")

    try:
        while True:
            # ดึงเฟรมภาพล่าสุด
            img = ep_camera.read_cv2_image(strategy="newest")
            
            if img is not None:
                cv2.imshow("Robot Live Stream", img)

            # รอรับการกดปุ่ม (1 ms)
            key = cv2.waitKey(1) & 0xFF

            # กดปุ่ม 's' เพื่อบันทึกภาพ
            if key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.jpg"
                cv2.imwrite(filename, img)
                print(f"[SUCCESS] บันทึกภาพแล้ว: {filename}")

            # กดปุ่ม 'q' เพื่อปิดโปรแกรม
            elif key == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()
        ep_camera.stop_video_stream()
        ep_robot.close()
        print("ปิดการเชื่อมต่อเรียบร้อย")