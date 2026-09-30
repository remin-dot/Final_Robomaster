import time

class ArmGripperController:
    def __init__(self, ep_robot):
        # เข้าถึงโมดูล robotic_arm และ gripper จากตัวหุ่นยนต์
        self.ep_arm = ep_robot.robotic_arm
        self.ep_gripper = ep_robot.gripper

    def pick_up_object(self, x_dist=40, y_dist=20, power=50):
        """
        ลำดับการทำงานสำหรับหยิบจับวัตถุ:
        1. เปิดมือจับ
        2. ยื่นแขนไปข้างหน้า/ลงล่าง
        3. ปิดมือจับเพื่อหยิบ
        4. ยกแขนขึ้น
        """
        print("--> กำลังเปิด Gripper...")
        self.ep_gripper.open(power=power)
        time.sleep(1)
        self.ep_gripper.pause()

        print(f"--> เลื่อนแขนกลไปที่ x={x_dist} mm, y={y_dist} mm...")
        self.ep_arm.move(x=x_dist, y=y_dist).wait_for_completed()

        print("--> กำลังปิด Gripper เพื่อหนีบวัตถุ...")
        self.ep_gripper.close(power=power)
        time.sleep(1)
        self.ep_gripper.pause()

        print("--> ยกแขนกลขึ้น...")
        self.ep_arm.move(x=0, y=180).wait_for_completed()

    def release_object(self, power=50):
        """ลำดับการทำงานสำหรับวางวัตถุ"""
        print("--> กำลังเปิด Gripper เพื่อปล่อยวัตถุ...")
        self.ep_arm.move(x=0, y=-120).wait_for_completed()
        self.ep_gripper.open(power=power)
        time.sleep(1)
        self.ep_gripper.pause()
        
    def reset_position(self, power=50):
        """เก็บแขนกลกลับตำแหน่งเริ่มต้น"""
        print("--> กำลังรีเซ็ตตำแหน่งแขนกล...")
        self.ep_arm.recenter().wait_for_completed()
        self.ep_gripper.close(power=power)
        time.sleep(1)
        self.ep_gripper.pause()