import sys
import time
import tkinter as tk
from robomaster import robot
from target_controller import (
    run_auto_shooting_mission,
    AVAILABLE_COLORS,
    AVAILABLE_SHAPES,
    SHAPE_TH_MAP,
    COLOR_TH_MAP
)

# ป้องกันปัญหาแสดงผลภาษาไทยบน Windows Terminal
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class MissionConfigDialog:
    """หน้าต่าง GUI ให้ผู้ใช้เลือกเป้าหมายที่ต้องการเล็งอย่างง่าย"""
    def __init__(self, default_color="Red", default_shape="Circle", default_shots=2):
        self.selected_color = default_color
        self.selected_shape = default_shape
        self.shots = default_shots
        self.enable_fire = True
        self.is_confirmed = False

        self.root = tk.Tk()
        self.root.title("🎯 ตั้งค่าภารกิจเล็งเป้าอัตโนมัติ (RoboMaster Mission Setup)")
        self.root.geometry("580x520")
        self.root.resizable(False, False)
        self.root.configure(bg="#1E1E2E")

        self.color_var = tk.StringVar(value=self.selected_color)
        self.shape_var = tk.StringVar(value=self.selected_shape)
        self.shots_var = tk.IntVar(value=self.shots)
        self.fire_var = tk.BooleanVar(value=True)

        self._build_ui()

    def _build_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#2A2A3E", padx=16, pady=12)
        header.pack(fill="x")

        tk.Label(header, text="🎯 เลือกระบบเล็งเป้าหมายอัตโนมัติ (Auto-Count & Double Shot)",
                 font=("Segoe UI", 13, "bold"), fg="#00FFAA", bg="#2A2A3E").pack(anchor="w")
        tk.Label(header, text="เลือกประเภทเป้าหมายที่ต้องการ -> กล้องนับจำนวนอัตโนมัติ -> ยิงเป้าละ 2 นัด จากซ้ายไปขวา",
                 font=("Tahoma", 9), fg="#B0B0C8", bg="#2A2A3E").pack(anchor="w", pady=(2, 0))

        content = tk.Frame(self.root, bg="#1E1E2E", padx=20, pady=12)
        content.pack(fill="both", expand=True)

        # 1. ปุ่มลัดเป้าหมายยอดนิยม
        quick_frame = tk.LabelFrame(content, text=" ⚡ เลือกประเภทเป้าหมายด่วน ",
                                    font=("Tahoma", 10, "bold"), fg="#FFDD00", bg="#252538", padx=10, pady=10)
        quick_frame.pack(fill="x", pady=(0, 10))

        presets = [
            ("🔴 วงกลม สีแดง", "Red", "Circle", "#FF4444"),
            ("🟢 วงกลม สีเขียว", "Green", "Circle", "#2ECC71"),
            ("🔵 วงกลม สีน้ำเงิน", "Blue", "Circle", "#3498DB"),
            ("🟡 วงกลม สีเหลือง", "Yellow", "Circle", "#F1C40F"),
            ("⬛ สี่เหลี่ยม สีแดง", "Red", "Square", "#FF6666"),
            ("🎯 ทุกเป้าหมาย (ทุกสี)", "ALL", "ALL", "#00FFAA"),
        ]

        grid = tk.Frame(quick_frame, bg="#252538")
        grid.pack(fill="x")
        for i, (label, c, s, col) in enumerate(presets):
            btn = tk.Button(grid, text=label, font=("Tahoma", 9, "bold"), fg=col, bg="#1E1E2E",
                            activebackground="#3A3A55", relief="groove", bd=2, pady=5, cursor="hand2",
                            command=lambda c_val=c, s_val=s: self._set_preset(c_val, s_val))
            btn.grid(row=i // 3, column=i % 3, padx=4, pady=4, sticky="ew")
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        grid.columnconfigure(2, weight=1)

        # 2. ปรับแต่งสีและรูปทรงอย่างอิสระ
        opt_box = tk.LabelFrame(content, text=" 🎨 หรือปรับแต่งตามต้องการ ",
                                font=("Tahoma", 10, "bold"), fg="#00E5FF", bg="#252538", padx=10, pady=8)
        opt_box.pack(fill="x", pady=(0, 10))

        row_c = tk.Frame(opt_box, bg="#252538")
        row_c.pack(fill="x", pady=3)
        tk.Label(row_c, text="สีเป้าหมาย:", font=("Tahoma", 9, "bold"), fg="white", bg="#252538", width=12, anchor="w").pack(side="left")
        for cid, cth, chex, _ in AVAILABLE_COLORS:
            tk.Radiobutton(row_c, text=cth, value=cid, variable=self.color_var,
                           font=("Tahoma", 9), fg=chex, bg="#252538", selectcolor="#14141E").pack(side="left", padx=3)

        row_s = tk.Frame(opt_box, bg="#252538")
        row_s.pack(fill="x", pady=3)
        tk.Label(row_s, text="รูปทรงเป้า:", font=("Tahoma", 9, "bold"), fg="white", bg="#252538", width=12, anchor="w").pack(side="left")
        for sid, sth, sicon in AVAILABLE_SHAPES:
            tk.Radiobutton(row_s, text=f"{sicon} {sth}", value=sid, variable=self.shape_var,
                           font=("Tahoma", 9), fg="#EAEAEA", bg="#252538", selectcolor="#14141E").pack(side="left", padx=3)

        # 3. ตัวเลือกจำนวนนัด และระบบยิง
        shot_frame = tk.Frame(content, bg="#1E1E2E")
        shot_frame.pack(fill="x", pady=5)

        tk.Checkbutton(shot_frame, text="💧 ยิงกระสุนเจลน้ำจริง (WATER_FIRE เท่านั้น - ไม่ใช้อินฟราเรด)", variable=self.fire_var,
                       font=("Tahoma", 9, "bold"), fg="#00FFAA", bg="#1E1E2E", selectcolor="#203040").pack(side="left")

        sub_shot = tk.Frame(shot_frame, bg="#1E1E2E")
        sub_shot.pack(side="right")
        tk.Label(sub_shot, text="ยิงเป้าละ:", font=("Tahoma", 9, "bold"), fg="#FFD700", bg="#1E1E2E").pack(side="left", padx=2)
        tk.Spinbox(sub_shot, from_=1, to=5, width=4, textvariable=self.shots_var,
                   font=("Tahoma", 10, "bold"), bg="#2A2A3E", fg="#00FFAA").pack(side="left")
        tk.Label(sub_shot, text="นัด (ค่าเริ่มต้น 2 นัด)", font=("Tahoma", 8), fg="#AAAAAA", bg="#1E1E2E").pack(side="left", padx=2)

        # 4. ปุ่มเริ่มภารกิจ
        btn_box = tk.Frame(self.root, bg="#1E1E2E", pady=10, padx=20)
        btn_box.pack(fill="x")

        start_btn = tk.Button(btn_box, text="🚀 เริ่มภารกิจ & สแกนนับเป้าหมายอัตโนมัติ (START)",
                              font=("Segoe UI", 12, "bold"), bg="#00C853", fg="white",
                              activebackground="#00E676", pady=7, cursor="hand2", relief="flat", command=self._on_start)
        start_btn.pack(side="left", fill="x", expand=True, padx=(0, 10))

        cancel_btn = tk.Button(btn_box, text="ออก (Exit)", font=("Tahoma", 10),
                               bg="#555566", fg="white", activebackground="#777788", pady=7, padx=14,
                               cursor="hand2", relief="flat", command=self._on_cancel)
        cancel_btn.pack(side="right")

    def _set_preset(self, color, shape):
        self.color_var.set(color)
        self.shape_var.set(shape)

    def _on_start(self):
        self.selected_color = self.color_var.get()
        self.selected_shape = self.shape_var.get()
        self.shots = max(1, min(5, self.shots_var.get()))
        self.enable_fire = self.fire_var.get()
        self.is_confirmed = True
        self.root.destroy()

    def _on_cancel(self):
        self.is_confirmed = False
        self.root.destroy()

    def run(self):
        self.root.mainloop()
        return self.selected_color, self.selected_shape, self.shots, self.enable_fire, self.is_confirmed


def main():
    print("="*80)
    print("  🚀 RoboMaster Autonomous Target Sequencing & Double-Shot System")
    print("  (ระบบเล็งเป้าหมายอัตโนมัติ ยิงเป้าละ 2 นัด จากซ้ายไปขวา)")
    print("="*80)

    # 1. แสดงหน้าต่างให้เลือกเป้าหมาย
    dialog = MissionConfigDialog(default_color="Red", default_shape="Circle", default_shots=2)
    target_color, target_shape, shots_per_target, enable_fire, confirmed = dialog.run()

    if not confirmed:
        print(">> ผู้ใช้กดยกเลิกภารกิจ -> ปิดโปรแกรม")
        return

    color_name_th = COLOR_TH_MAP.get(target_color, target_color)
    shape_name_th = SHAPE_TH_MAP.get(target_shape, target_shape)
    print(f"\n📋 [เป้าหมายที่เลือก]: {color_name_th} - {shape_name_th}")
    print(f"💥 [ตั้งค่าการยิง]: เป้าละ {shots_per_target} นัด | ยิงจริง: {enable_fire}")

    # 2. เชื่อมต่อหุ่นยนต์ RoboMaster EP
    print("\nกำลังเชื่อมต่อกับหุ่นยนต์ RoboMaster EP (โหมด AP)...")
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    try:
        # 3. เริ่มเปิดกล้อง
        ep_robot.camera.start_video_stream(display=False)
        time.sleep(1.0)

        # 4. เรียกใช้ฟังก์ชันหลักในการรันภารกิจ
        run_auto_shooting_mission(
            ep_robot=ep_robot,
            color_name=target_color,
            shape_name=target_shape,
            shots_per_target=shots_per_target,
            enable_fire=enable_fire
        )

    except KeyboardInterrupt:
        print("\n[!] ผู้ใช้สั่งหยุดการทำงานด้วยแป้นพิมพ์ (Ctrl+C)")
    except Exception as e:
        print(f"\n[!] เกิดข้อผิดพลาดระหว่างทำงาน: {e}")
    finally:
        # 5. ปิดอุปกรณ์อย่างปลอดภัย
        print("กำลังปิดการเชื่อมต่อหุ่นยนต์...")
        try:
            ep_robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
            ep_robot.camera.stop_video_stream()
            ep_robot.close()
        except Exception:
            pass
        print(">> สิ้นสุดการทำงานอย่างปลอดภัย เรียบร้อยแล้ว")


if __name__ == "__main__":
    main()
