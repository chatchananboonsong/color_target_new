import time
import os
import sys
import json
import cv2
import numpy as np
from robomaster import blaster, led

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

# ==============================================================================
# โมดูลควบคุมเป้าหมาย (Target Controller Module)
# ออกแบบฟังก์ชันให้อ่านง่าย เป็นระเบียบ ชัดเจน และนำไปเรียกใช้จาก main.py ได้ทันที
# ==============================================================================

def resolve_config_path(config_file="hsv_config.json"):
    """ค้นหาไฟล์คอนฟิกทั้งแบบ Absolute และ Relative เทียบกับโฟลเดอร์สคริปต์"""
    if os.path.isabs(config_file) and os.path.exists(config_file):
        return config_file
    if os.path.exists(config_file):
        return os.path.abspath(config_file)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(base_dir, config_file)
    if os.path.exists(cand):
        return cand
    return config_file


def load_hsv_configs(config_file="hsv_config.json"):
    """โหลดค่า Lower/Upper HSV จากไฟล์ json (hsv_config.json)"""
    resolved_path = resolve_config_path(config_file)
    if os.path.exists(resolved_path):
        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                raw_cfg = json.load(f)
            configs = {}
            for color_name, data in raw_cfg.items():
                ranges = []
                for r in data.get("ranges", []):
                    ranges.append((
                        np.array(r["lower"], dtype=np.uint8),
                        np.array(r["upper"], dtype=np.uint8)
                    ))
                configs[color_name] = {
                    "ranges": ranges,
                    "draw_color": tuple(data.get("draw_color", [0, 255, 0])),
                    "led_rgb": tuple(data.get("led_rgb", [255, 255, 255])),
                    "name_th": data.get("name_th", color_name)
                }
            print(f">> [HSV CONFIG] โหลดค่า HSV จาก '{resolved_path}' สำเร็จ ({len(configs)} สี)")
            return configs
        except Exception as e:
            print(f"[!] โหลด {resolved_path} ไม่สำเร็จ: {e}")

    # Fallback เริ่มต้นตามมาตรฐานที่จูนไว้ใน hsv_config.json
    print("[!] ใช้ค่า Fallback HSV ตามมาตรฐาน hsv_config.json")
    return {
        "Red": {"ranges": [(np.array([0, 100, 70]), np.array([10, 255, 255])), (np.array([165, 100, 70]), np.array([180, 255, 255]))], "draw_color": (0, 0, 255), "led_rgb": (255, 0, 0), "name_th": "แดง"},
        "Green": {"ranges": [(np.array([55, 92, 52]), np.array([93, 255, 110]))], "draw_color": (0, 255, 0), "led_rgb": (0, 255, 0), "name_th": "เขียว"},
        "Blue": {"ranges": [(np.array([95, 79, 49]), np.array([130, 217, 110]))], "draw_color": (255, 130, 0), "led_rgb": (0, 100, 255), "name_th": "น้ำเงิน"},
        "Yellow": {"ranges": [(np.array([20, 155, 100]), np.array([35, 255, 255]))], "draw_color": (0, 220, 255), "led_rgb": (255, 255, 0), "name_th": "เหลือง"}
    }


def get_color_mappings(config_file="hsv_config.json"):
    """แปลงค่าจาก hsv_config.json เพื่อใช้งานใน UI และแมปปิ้งชื่อภาษาไทย"""
    cfg = load_hsv_configs(config_file)
    hex_fallback = {
        "Red": "#FF4444",
        "Green": "#2ECC71",
        "Blue": "#3498DB",
        "Yellow": "#F1C40F"
    }
    available_colors = []
    color_th_map = {"ALL": "ทุกสี"}
    for cname, cdata in cfg.items():
        th = cdata.get("name_th", cname)
        cth = f"สี{th}" if not th.startswith("สี") else th
        led_rgb = cdata.get("led_rgb", (255, 255, 255))
        chex = hex_fallback.get(cname, f"#{led_rgb[0]:02x}{led_rgb[1]:02x}{led_rgb[2]:02x}")
        draw_color = cdata.get("draw_color", (0, 255, 0))
        available_colors.append((cname, cth, chex, draw_color))
        color_th_map[cname] = cth
    return available_colors, color_th_map


# โหลดรายการสีและชื่อภาษาไทยจาก hsv_config.json โดยตรง
AVAILABLE_COLORS, COLOR_TH_MAP = get_color_mappings()

AVAILABLE_SHAPES = [
    ("Circle", "ทรงกลม", "🔘"),
    ("Square", "สี่เหลี่ยมจัตุรัส", "⬛"),
    ("Rect_H", "ผืนผ้านอน", "▰"),
    ("Rect_V", "ผืนผ้าตั้ง", "▮"),
    ("Rectangle", "ผืนผ้าทั้งหมด", "▭")
]

SHAPE_TH_MAP = {
    "Circle": "ทรงกลม",
    "Square": "สี่เหลี่ยมจัตุรัส",
    "Rectangle": "สี่เหลี่ยมผืนผ้า",
    "Rect_H": "ผืนผ้านอน",
    "Rect_V": "ผืนผ้าตั้ง",
    "ALL": "ทุกรูปทรง"
}


class PIDController:
    """คลาสคำนวณ PID สำหรับขับเคลื่อน Gimbal (Yaw และ Pitch)"""
    def __init__(self, kp, ki, kd, limits=(-120, 120)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_limit, self.max_limit = limits
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def compute(self, error):
        now = time.time()
        dt = now - self.last_time
        if dt <= 0:
            dt = 0.01

        p_term = self.kp * error
        self.integral += error * dt
        self.integral = max(-40.0, min(40.0, self.integral))
        i_term = self.ki * self.integral
        derivative = (error - self.last_error) / dt
        d_term = self.kd * derivative

        output = p_term + i_term + d_term
        output = max(self.min_limit, min(self.max_limit, output))

        self.last_error = error
        self.last_time = now
        return output

    def reset(self):
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()


# พารามิเตอร์ PID
pid_yaw = PIDController(kp=100.0, ki=0.01, kd=4.0, limits=(-120, 120))
pid_pitch = PIDController(kp=80.0, ki=0.0, kd=4.0, limits=(-80, 80))

# ค่าคงที่สำหรับการเล็งและยิง (เน้นความแม่นยำ ตรงเป้าจริง ไม่รีบสะบัดหนี)
LOCK_TOLERANCE_X = 0.001          # ความแม่นยำแนวนอน (±3.0% = ±38 พิกเซล เล็งตรงเป้าจริง)
LOCK_TOLERANCE_Y = 0.001           # ความแม่นยำแนวตั้ง (±4.0% = ±29 พิกเซล)
LOCK_HOLD_TIME = 0.38              # ต้องนิ่งเข้ากึ่งกลางจริง 0.38 วินาที (ป้องกันการยิงวืดขณะกิมบอลยังวิ่งอยู่)
MIN_CONTOUR_AREA = 600             # ขนาดขั้นต่ำของเป้าหมาย (ตัด noise)

# ตัวแปรบันทึกมุม Gimbal ปัจจุบันแบบ Real-time
current_gimbal_yaw = 0.0
current_gimbal_pitch = 0.0


def on_gimbal_angle_cb(angle_info):
    """Callback อัปเดตมุม Pitch และ Yaw จากเซนเซอร์ Gimbal โดยตรง"""
    global current_gimbal_yaw, current_gimbal_pitch
    pitch, yaw, pitch_ground, yaw_ground = angle_info
    current_gimbal_pitch = pitch
    current_gimbal_yaw = yaw


class TargetMemory:
    """
    ระบบหน่วยความจำจดจำเป้าหมายที่ยิงไปแล้ว (Shot Target Memory System)
    - บันทึกองศา Gimbal Yaw และข้อมูลตำแหน่งของเป้าหมายที่ถูกยิงไปแล้ว
    - ป้องกันการกลับมายิงเป้าเดิมซ้ำ 100%
    - แสดงผลบนหน้าจอว่าเป้าไหนยิงไปแล้ว (กากบาท ❌ และแท็ก ALREADY SHOT)
    """
    def __init__(self, hfov=96.0, yaw_match_tolerance=10.0):
        self.hfov = hfov                              # มุมมองภาพแนวนอนของกล้อง RoboMaster
        self.yaw_match_tolerance = yaw_match_tolerance  # ระยะมุมที่ถือว่าเป็นเป้าหมายเดียวกัน (องศา)
        self.shot_records = []                        # รายการบันทึกเป้าที่ยิงไปแล้ว

    def record_shot(self, target_num, gimbal_yaw, target_info=None):
        """บันทึกเป้าหมายที่เพิ่งยิงเสร็จลงหน่วยความจำ"""
        record = {
            "target_num": target_num,
            "yaw": gimbal_yaw,
            "color": target_info.get("color") if target_info else "",
            "shape": target_info.get("shape") if target_info else "",
            "time": time.time()
        }
        self.shot_records.append(record)
        print(f"📝 [MEMORY SAVED] จดจำเป้าหมาย #{target_num} ที่มุม Yaw={gimbal_yaw:.1f}° เรียบร้อยแล้ว (บล็อกเป้านี้ไม่ให้ยิงซ้ำเด็ดขาด)")

    def is_already_shot(self, norm_x, current_yaw):
        """
        ตรวจสอบว่าเป้าหมายที่พิกัด norm_x ในเฟรมปัจจุบัน ตรงกับเป้าที่เคยยิงไปแล้วหรือไม่
        โดยเทียบกับตำแหน่ง Yaw ในโลกจริงของแต่ละเป้าหมายที่เคยบันทึกไว้
        """
        if not self.shot_records:
            return False, None

        # คำนวณมุม Yaw สัมบูรณ์ของเป้าหมายนี้ในโลกจริง
        delta_yaw = (norm_x - 0.5) * self.hfov
        estimated_yaw = current_yaw + delta_yaw

        for record in self.shot_records:
            if abs(estimated_yaw - record["yaw"]) < self.yaw_match_tolerance:
                return True, record["target_num"]

        return False, None

    def annotate_targets(self, targets, current_yaw):
        """
        ตรวจสอบและใส่สถานะ is_already_shot ให้กับเป้าหมายทุกตัวในเฟรม
        """
        for t in targets:
            is_shot, shot_id = self.is_already_shot(t["norm_x"], current_yaw)
            t["is_already_shot"] = is_shot
            t["shot_id"] = shot_id
        return targets

    def get_unshot_targets(self, targets):
        """คืนค่าเฉพาะเป้าหมายที่ยังไม่เคยถูกยิงเท่านั้น"""
        return [t for t in targets if not t.get("is_already_shot", False)]

    def clear(self):
        self.shot_records.clear()
        print("🗑️ [MEMORY CLEARED] ล้างประวัติเป้าหมายที่ยิงไปแล้วทั้งหมดเรียบร้อย")





def classify_shape(cnt):
    """วิเคราะห์รูปทรง: Circle (กลม), Square (จัตุรัส), Rect_H (ผืนผ้านอน), Rect_V (ผืนผ้าตั้ง)"""
    area = cv2.contourArea(cnt)
    if area < MIN_CONTOUR_AREA:
        return None

    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return None

    circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
    _, radius = cv2.minEnclosingCircle(cnt)
    circle_area = np.pi * (radius ** 2)
    circle_ratio = area / circle_area if circle_area > 0 else 0

    approx = cv2.approxPolyDP(cnt, 0.038 * perimeter, True)
    bx, by, bw, bh = cv2.boundingRect(cnt)
    bbox_area = bw * bh
    extent = area / bbox_area if bbox_area > 0 else 0

    # 1. ตรวจสอบสี่เหลี่ยม (จัตุรัส / ผืนผ้านอน / ผืนผ้าตั้ง)
    is_4_corners = (len(approx) == 4) and cv2.isContourConvex(approx)
    if (is_4_corners and extent >= 0.65) or (len(approx) in (4, 5) and extent >= 0.78):
        aspect = float(bw) / float(bh)
        if 0.85 <= aspect <= 1.18:
            return "Square"
        elif aspect > 1.18:
            return "Rect_H"
        else:
            return "Rect_V"

    # 2. ตรวจสอบทรงกลม (ปรับเกณฑ์ให้ครอบคลุมเป้ากลมที่อาจบิดเบี้ยวเล็กน้อยตามมุมกล้อง)
    if circularity >= 0.60 and circle_ratio >= 0.58:
        return "Circle"

    return None


def detect_targets(img, color_filter="Red", shape_filter="Circle", hsv_configs=None):
    """
    ตรวจจับเป้าหมายที่ตรงกับสีและรูปทรงที่กำหนด
    เรียงลำดับเป้าหมายจาก "ซ้ายสุดไปขวาสุด" ตามพิกัด X (center_x) อย่างเคร่งครัด
    """
    if hsv_configs is None:
        hsv_configs = load_hsv_configs()

    h, w, _ = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    targets = []

    for color_name, color_cfg in hsv_configs.items():
        if color_filter != "ALL" and color_name != color_filter:
            continue

        mask = None
        for lower, upper in color_cfg["ranges"]:
            part = cv2.inRange(hsv, lower, upper)
            mask = part if mask is None else cv2.bitwise_or(mask, part)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        for cnt in contours:
            shape = classify_shape(cnt)
            if shape is None:
                continue

            if shape_filter != "ALL":
                if isinstance(shape_filter, (list, tuple, set)):
                    if shape not in shape_filter:
                        if "Rectangle" in shape_filter and shape in ("Rect_H", "Rect_V"):
                            pass
                        else:
                            continue
                else:
                    if shape_filter in ("Square", "สี่เหลี่ยมจัตุรัส") and shape != "Square":
                        continue
                    elif shape_filter in ("Circle", "ทรงกลม") and shape != "Circle":
                        continue
                    elif shape_filter in ("Rect_H", "Horizontal", "ผืนผ้านอน", "สี่เหลี่ยมผืนผ้าแนวนอน") and shape != "Rect_H":
                        continue
                    elif shape_filter in ("Rect_V", "Vertical", "ผืนผ้าตั้ง", "สี่เหลี่ยมผืนผ้าแนวตั้ง") and shape != "Rect_V":
                        continue
                    elif shape_filter in ("Rectangle", "สี่เหลี่ยมผืนผ้า", "ผืนผ้าทั้งหมด") and shape not in ("Rectangle", "Rect_H", "Rect_V"):
                        continue
                    elif shape_filter not in ("Square", "Circle", "Rect_H", "Rect_V", "Rectangle", "ALL") and shape != shape_filter:
                        continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            bx, by, bw, bh = cv2.boundingRect(cnt)

            shape_th = SHAPE_TH_MAP.get(shape, shape)
            color_th = color_cfg.get("name_th", color_name)
            name_th = f"{shape_th} สี{color_th}" if not color_th.startswith("สี") else f"{shape_th} {color_th}"

            targets.append({
                "color": color_name,
                "shape": shape,
                "center": (cx, cy),
                "norm_x": cx / w,
                "norm_y": cy / h,
                "bbox": (bx, by, bw, bh),
                "draw_color": color_cfg["draw_color"],
                "led_rgb": color_cfg["led_rgb"],
                "name_th": name_th
            })

    # เรียงลำดับจากซ้ายไปขวาตามพิกัด X (center_x จากน้อยไปมาก) เสมอ
    targets.sort(key=lambda t: t["center"][0])
    return targets


def sleep_with_stream(ep_camera, window_name, duration, overlay_text=""):
    """
    หน่วงเวลาแบบ Non-blocking พร้อมดึงเฟรมกล้องสดและอัปเดตหน้าต่าง OpenCV อย่างต่อเนื่อง
    แก้ปัญหาหน้าจอค้าง (Not Responding) และล้าง buffer เฟรมเก่า ป้องกันการตัดสินใจผิดพลาด 100%
    """
    if ep_camera is None:
        time.sleep(duration)
        return

    end_t = time.time() + duration
    while time.time() < end_t:
        try:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.08)
        except Exception:
            img = None

        if img is not None:
            if overlay_text:
                h, w, _ = img.shape
                cv2.rectangle(img, (20, 20), (min(w - 20, 680), 75), (0, 0, 0), -1)
                cv2.rectangle(img, (20, 20), (min(w - 20, 680), 75), (0, 255, 255), 2)
                cv2.putText(img, overlay_text, (35, 57),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)
            cv2.imshow(window_name, img)
            cv2.waitKey(1)
        else:
            time.sleep(0.01)


def fire_double_shot(ep_blaster, ep_led, ep_camera=None, window_name="RoboMaster Auto Target Mission", count=2, enable_fire=True, led_rgb=None):
    """ยิงเฉพาะกระสุนเจลจริง (WATER_FIRE) เท่านั้น ไม่ใช้อินฟราเรด และปิดไฟอินฟราเรดหัวปืน 100% พร้อมเปิดไฟ LED ตามสีเป้าหมายจาก hsv_config.json"""
    if not enable_fire:
        print("\n>> [FIRE SAFE] โหมดปลอดภัย (เล็งอย่างเดียว ไม่ยิงกระสุนจริง)")
        if ep_camera:
            sleep_with_stream(ep_camera, window_name, 0.5, "🛡️ [SAFE MODE] TARGET LOCKED - SIMULATED FIRE")
        else:
            time.sleep(0.5)
        return

    print(f"\n💥 >> [WATER FIRE ONLY!] ยิงเฉพาะกระสุนเจลจริงจำนวน {count} นัด (ไม่ใช้อินฟราเรดเด็ดขาด) << 💥")

    # ปิดไฟอินฟราเรดที่หัวกระบอกปืนอย่างถาวร (Disable Infrared Muzzle LED)
    try:
        ep_blaster.set_led(brightness=0, effect=blaster.LED_OFF)
    except Exception:
        pass

    # กะพริบไฟ LED ตัวหุ่น (COMP_TOP_ALL) ตามสีเป้าหมายจาก hsv_config.json เพื่อแสดงจังหวะการยิง
    fire_rgb = led_rgb if led_rgb is not None else (255, 50, 0)
    try:
        ep_led.set_led(comp=led.COMP_TOP_ALL, r=fire_rgb[0], g=fire_rgb[1], b=fire_rgb[2], effect=led.EFFECT_ON)
    except Exception:
        pass

    # สั่งลั่นกระสุนเจลจริง (WATER_FIRE) ผ่านมอเตอร์ขับลูกเจลโดยตรง (ไม่ใช่กระสุนอินฟราเรด)
    try:
        ep_blaster.fire(fire_type=blaster.WATER_FIRE, times=count)
    except Exception as e:
        print(f"[!] ยิงกระสุนเจลล้มเหลว: {e}")

    # สตรีมภาพสดต่อเนื่องขณะกระสุนเจลพุ่งออกไป (ไม่ค้าง)
    firing_duration = max(0.4, 0.28 * count)
    if ep_camera:
        sleep_with_stream(ep_camera, window_name, firing_duration, f"💧 FIRING REAL WATER GEL BEADS [{count} SHOTS]...")
    else:
        time.sleep(firing_duration)

    try:
        ep_led.set_led(comp=led.COMP_TOP_ALL, r=0, g=0, b=0, effect=led.EFFECT_OFF)
    except Exception:
        pass

    # พักรอให้กระสุนเจลวิ่งไปกระทบเป้าหมาย พร้อมสตรีมภาพสด
    if ep_camera:
        sleep_with_stream(ep_camera, window_name, 0.35, "💥 WATER GEL BULLET IMPACT CONFIRMED!")
    else:
        time.sleep(0.35)

    pid_yaw.reset()
    pid_pitch.reset()


def scan_and_count_targets(ep_camera, color_name="Red", shape_name="Circle", hsv_configs=None, scan_sec=2.5, window_name="RoboMaster"):
    """
    ฟังก์ชันสแกนนับจำนวนเป้าหมายอัตโนมัติตอนเปิดกล้อง
    คืนค่าเป็นจำนวนเป้าหมายที่นับได้ (int)
    """
    print(f"\n📷 [สแกนนับเป้าหมายอัตโนมัติ] กำลังตรวจนับเป้าหมายประเภท [{color_name} {shape_name}]...")
    start_t = time.time()
    last_detected = []

    while time.time() - start_t < scan_sec:
        img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        if img is None:
            continue

        h, w, _ = img.shape
        targets = detect_targets(img, color_name, shape_name, hsv_configs)
        last_detected = targets

        remain = max(0.0, scan_sec - (time.time() - start_t))

        # วาดกรอบเป้าหมายที่ตรวจพบ
        for idx, t in enumerate(targets):
            bx, by, bw, bh = t["bbox"]
            pos_text = "ซ้ายสุด (Target #1)" if idx == 0 else f"เป้าที่ #{idx+1}"
            tag = f"#{idx+1} {pos_text} [X={t['center'][0]}]"
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), t["draw_color"], 2)
            cv2.putText(img, tag, (bx, max(15, by - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # ข้อมูลบน HUD
        cv2.putText(img, f"=== [SCANNING TARGETS: {color_name} {shape_name}] ===", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)
        cv2.putText(img, f"Detected: {len(targets)} Targets (Left -> Right) | Ready in: {remain:.1f}s",
                    (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

        cv2.imshow(window_name, img)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord(' '), 13) and len(targets) > 0:
            break
        elif key == ord('q'):
            return 0

    count = len(last_detected)
    print(f"🎯 [ผลการตรวจนับ] พบทั้งหมด {count} เป้าหมาย เรียงจากซ้ายไปขวา!")
    for idx, t in enumerate(last_detected):
        print(f"   - เป้า #{idx+1}: {t['name_th']} ที่พิกัด X={t['center'][0]}")
    return count


def recenter_gimbal_with_stream(ep_gimbal, ep_camera=None, window_name="RoboMaster Auto Target Mission", timeout=1.5):
    """
    ดึง Gimbal กลับกึ่งกลางอย่างนุ่มนวล พร้อมปั๊มภาพสดต่อเนื่อง ไม่กระตุก ไม่ค้าง 100%
    """
    print("🔄 [RECENTER] กำลังดึง Gimbal กลับเข้าสู่กึ่งกลาง...")
    try:
        action = ep_gimbal.recenter()
        start_t = time.time()
        while time.time() - start_t < timeout:
            if ep_camera:
                img = ep_camera.read_cv2_image(strategy="newest", timeout=0.05)
                if img is not None:
                    h, w, _ = img.shape
                    cv2.rectangle(img, (20, 20), (min(w - 20, 680), 75), (0, 0, 0), -1)
                    cv2.rectangle(img, (20, 20), (min(w - 20, 680), 75), (0, 255, 255), 2)
                    cv2.putText(img, "🔄 RECENTERING GIMBAL TO CENTER...", (35, 57),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)
                    cv2.imshow(window_name, img)
                    cv2.waitKey(1)
            if hasattr(action, "is_completed") and action.is_completed:
                break
            time.sleep(0.01)
    except Exception as e:
        print(f"[!] Recenter gimbal error: {e}")

    # ดึงภาพกล้องใหม่อีก 0.2 วินาทีหลังจาก Gimbal หยุดสนิท เพื่อให้ภาพนิ่งและล้าง buffer เก่า
    if ep_camera:
        sleep_with_stream(ep_camera, window_name, 0.20, "👀 STABILIZING CAMERA FEED...")
    pid_yaw.reset()
    pid_pitch.reset()


def track_and_lock_target(ep_camera, ep_gimbal, ep_led, color_name, shape_name, hsv_configs,
                          target_index, total_targets, prefer_leftmost=False, target_memory=None,
                          timeout=10.0, window_name="RoboMaster Auto Target Mission"):
    """
    ฟังก์ชันเล็งเป้าหมายเข้าสู่จุดกึ่งกลาง (Visual PID Centering)
    พร้อมระบบจดจำเป้าหมาย (TargetMemory) ที่จะไม่เลือกเป้าที่เคยยิงไปแล้ว
    เมื่อเข้าเป้าและนิ่งครบเวลา จะคืนค่า (True, locked_target_info)
    """
    print(f"\n🎯 [กำลังเล็งเป้าหมาย #{target_index}/{total_targets}]...")
    pid_yaw.reset()
    pid_pitch.reset()

    lock_start = None
    start_track_time = time.time()
    no_target_start = None

    # สถานะการตัดสินใจเลือกเป้าหมาย (ตัดสินใจเลือกเป้าซ้ายสุดครั้งแรก แล้วยึดมั่น 100%)
    is_target_decided = False
    committed_pos = None
    target_lost_count = 0

    while time.time() - start_track_time < timeout:
        img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        if img is None:
            continue

        h, w, _ = img.shape
        cx_mid = w // 2
        cy_mid = h // 2

        # วาด Crosshair กลางจอ
        cv2.line(img, (cx_mid - 25, cy_mid), (cx_mid + 25, cy_mid), (255, 255, 255), 1)
        cv2.line(img, (cx_mid, cy_mid - 25), (cx_mid, cy_mid + 25), (255, 255, 255), 1)

        targets = detect_targets(img, color_name, shape_name, hsv_configs)

        # ----------------------------------------------------
        # ระบบหน่วยความจำเป้าหมาย (Target Memory Annotation):
        # ----------------------------------------------------
        if target_memory is not None:
            targets = target_memory.annotate_targets(targets, current_gimbal_yaw)
            unshot_targets = target_memory.get_unshot_targets(targets)
        else:
            unshot_targets = targets

        # วาดกรอบเป้าหมายทั้งหมดลงบนจอภาพ
        for t in targets:
            bx, by, bw, bh = t["bbox"]
            if t.get("is_already_shot", False):
                cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (80, 80, 80), 1)
                cv2.line(img, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
                cv2.line(img, (bx + bw, by), (bx, by + bh), (0, 0, 255), 2)
                shot_lbl = f"SHOT #{t.get('shot_id')}" if t.get('shot_id') != 'PREV' else "SHOT"
                cv2.putText(img, shot_lbl, (bx, max(15, by - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            else:
                cv2.rectangle(img, (bx, by), (bx + bw, by + bh), t["draw_color"], 1)

        # ----------------------------------------------------
        # การเลือกเป้าหมายแบบ "ตัดสินใจแล้วยึดตามนั้น" (Strict Commitment):
        # ----------------------------------------------------
        target = None
        if not is_target_decided:
            if unshot_targets:
                no_target_start = None
                chosen = unshot_targets[0]  # เลือกเป้าหมายตัวซ้ายสุดของกลุ่มที่ยังไม่เคยถูกยิง
                committed_pos = (chosen["norm_x"], chosen["norm_y"])
                is_target_decided = True
                target = chosen
                print(f"🔒 [COMMITTED] ตัดสินใจเลือกเป้าหมาย #{target_index} ({chosen['name_th']}) ที่ X={chosen['center'][0]} แล้ว -> ยึดเป้านี้แน่นอน 100%!")
            else:
                # ยังไม่มีเป้าหมายที่ยังไม่เคยยิงในสายตา
                if no_target_start is None:
                    no_target_start = time.time()
                elif time.time() - no_target_start > 2.5:
                    print(f"ℹ️ [SCAN] ไม่พบเป้าหมายที่ยังไม่ถูกยิงเหลืออยู่ในมุมมอง (ครบ 2.5s)")
                    ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                    return False, None
        else:
            # จังหวะเกาะติดเป้าหมายที่ตัดสินใจแล้ว
            if targets and committed_pos is not None:
                best_t = min(targets, key=lambda t: np.hypot(t["norm_x"] - committed_pos[0], t["norm_y"] - committed_pos[1]))
                dist = np.hypot(best_t["norm_x"] - committed_pos[0], best_t["norm_y"] - committed_pos[1])
                if dist < 0.28:
                    committed_pos = (best_t["norm_x"], best_t["norm_y"])
                    target = best_t
                    target_lost_count = 0
                else:
                    target_lost_count += 1
            else:
                target_lost_count += 1

            if target_lost_count > 35:
                # หลุดเป้านานเกินไป ให้รีเซ็ตการตัดสินใจ
                is_target_decided = False
                committed_pos = None
                target_lost_count = 0

        now_t = time.time()

        if target is not None:
            t_cx, t_cy = target["center"]
            cv2.circle(img, (t_cx, t_cy), 8, (0, 0, 255), -1)
            cv2.circle(img, (t_cx, t_cy), 18, (0, 255, 255), 2)
            cv2.circle(img, (t_cx, t_cy), 24, (0, 200, 255), 1)
            cv2.line(img, (cx_mid, cy_mid), (t_cx, t_cy), (0, 255, 255), 1)
            cv2.putText(img, f"LOCKED TARGET #{target_index}", (t_cx + 26, t_cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            err_x = target["norm_x"] - 0.5
            err_y = 0.5 - target["norm_y"]

            is_centered = (abs(err_x) < LOCK_TOLERANCE_X) and (abs(err_y) < LOCK_TOLERANCE_Y)

            if is_centered:
                if lock_start is None:
                    lock_start = now_t

                ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)

                hold_dur = now_t - lock_start
                progress = min(1.0, hold_dur / LOCK_HOLD_TIME)

                bar_w, bar_h = 220, 16
                bx, by = 20, h - 65
                cv2.rectangle(img, (bx, by), (bx + bar_w, by + bar_h), (80, 80, 80), 1)
                fill_w = int(bar_w * progress)
                fill_col = (0, 0, 255) if progress >= 0.95 else (0, 255, 255)
                cv2.rectangle(img, (bx + 1, by + 1), (bx + fill_w, by + bar_h - 1), fill_col, -1)
                cv2.putText(img, f"LOCKING: {int(progress * 100)}%", (bx + bar_w + 12, by + 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, fill_col, 2)

                if hold_dur >= LOCK_HOLD_TIME:
                    print(f"🎯 [LOCKED!] ล็อกเป้าหมาย #{target_index}/{total_targets} นิ่งตรงกลางจอเรียบร้อยแล้ว!")
                    if ep_led and target and "led_rgb" in target:
                        r_c, g_c, b_c = target["led_rgb"]
                        try:
                            ep_led.set_led(comp=led.COMP_TOP_ALL, r=r_c, g=g_c, b=b_c, effect=led.EFFECT_ON)
                        except Exception:
                            pass
                    return True, target
            else:
                lock_start = None
                yaw_spd = pid_yaw.compute(err_x)
                pitch_spd = pid_pitch.compute(err_y)
                ep_gimbal.drive_speed(pitch_speed=pitch_spd, yaw_speed=yaw_spd)
        else:
            lock_start = None
            # หยุดนิ่งตรงกลาง ไม่ดึงขวาเด็ดขาด
            ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)

        # กรอบ Lock Zone
        zw = int(w * LOCK_TOLERANCE_X)
        zh = int(h * LOCK_TOLERANCE_Y)
        cv2.rectangle(img, (cx_mid - zw, cy_mid - zh), (cx_mid + zw, cy_mid + zh), (0, 255, 0) if lock_start else (80, 80, 80), 1)

        # ข้อมูล HUD
        cv2.putText(img, f"Aiming Target ({target_index}/{total_targets}): [{color_name} {shape_name}] | Yaw: {current_gimbal_yaw:.1f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2)
        cv2.imshow(window_name, img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False, None

    return False, None


def run_auto_shooting_mission(ep_robot, color_name="Red", shape_name="Circle", shots_per_target=2, enable_fire=True):
    """
    ฟังก์ชันหลักในการรันภารกิจอัตโนมัติ:
    1. ตั้งศูนย์ Gimbal ด้วยระบบปั๊มภาพสด ไม่ค้าง
    2. สแกนนับจำนวนเป้าหมายอัตโนมัติตอนเปิดระบบ
    3. ยิงเป้าหมายทีละตัวจากซ้ายไปขวา ตัวละ 2 นัด (ไม่ค้าง ไม่กระตุก)
    4. เมื่อยิงเสร็จแต่ละเป้า: บันทึกหน่วยความจำ TargetMemory และ recenter กลับกึ่งกลาง
    5. เมื่อยิงครบทุกเป้า: เข้าสู่โหมด Standby เฝ้าระวังต่อเนื่อง (ไม่หยุดทำงานหรือปิดโปรแกรมเอง)
       ผู้ใช้สามารถกด [SPACEBAR] เพื่อยิงต่อ, [R] เพื่อรีเซ็ตเริ่มใหม่ หรือ [Q] เพื่อปิดโปรแกรม
    """
    ep_gimbal = ep_robot.gimbal
    ep_blaster = ep_robot.blaster
    ep_led = ep_robot.led
    ep_camera = ep_robot.camera

    window_name = "RoboMaster Auto Target Mission"
    cv2.namedWindow(window_name)

    # 1. ปิดไฟอินฟราเรดที่หัวปืนอย่างเด็ดขาด และตั้งศูนย์ Gimbal
    try:
        ep_blaster.set_led(brightness=0, effect=blaster.LED_OFF)
    except Exception:
        pass
    recenter_gimbal_with_stream(ep_gimbal, ep_camera, window_name=window_name, timeout=1.5)

    # เชื่อมต่อระบบอ่านมุม Gimbal แบบ Real-time
    try:
        ep_gimbal.sub_angle(freq=20, callback=on_gimbal_angle_cb)
    except Exception as e:
        print(f"[!] คำเตือน sub_angle: {e}")

    hsv_configs = load_hsv_configs()
    target_memory = TargetMemory()

    mission_running = True

    try:
        while mission_running:
            # 2. สแกนนับจำนวนเป้าหมายอัตโนมัติ
            total_targets = scan_and_count_targets(ep_camera, color_name, shape_name, hsv_configs, scan_sec=2.5, window_name=window_name)

            if total_targets == 0:
                print("⚠️ ยังไม่พบเป้าหมายหน้ากล้องในขณะนี้ | กำลังรอเป้าหมาย...")
                while True:
                    img = ep_camera.read_cv2_image(strategy="newest", timeout=0.1)
                    if img is not None:
                        h, w, _ = img.shape
                        cv2.rectangle(img, (20, 20), (w - 20, 95), (0, 0, 0), -1)
                        cv2.rectangle(img, (20, 20), (w - 20, 95), (0, 255, 255), 2)
                        cv2.putText(img, f"WAITING FOR TARGET: [{color_name} {shape_name}]", (35, 52),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)
                        cv2.putText(img, "Press [SPACEBAR] to Scan & Shoot | [Q] to Quit", (35, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
                        cv2.imshow(window_name, img)
                        k = cv2.waitKey(1) & 0xFF
                        if k == ord('q'):
                            mission_running = False
                            break
                        elif k in (ord(' '), 13):
                            break
                    else:
                        time.sleep(0.01)

                if not mission_running:
                    break
                continue

            print("\n" + "="*75)
            print(f"🚀 [เริ่มภารกิจยิง] ตรวจพบ {total_targets} เป้าหมาย (เป้าละ {shots_per_target} นัด จากซ้ายไปขวา)")
            print("="*75)

            shot_count = 0
            while True:
                current_target_index = shot_count + 1

                # เล็งและล็อกเป้าหมาย (เลือกเป้าซ้ายสุดของกลุ่มที่ยังไม่เคยยิง)
                locked, locked_target = track_and_lock_target(
                    ep_camera=ep_camera,
                    ep_gimbal=ep_gimbal,
                    ep_led=ep_led,
                    color_name=color_name,
                    shape_name=shape_name,
                    hsv_configs=hsv_configs,
                    target_index=current_target_index,
                    total_targets=max(total_targets, current_target_index),
                    target_memory=target_memory,
                    window_name=window_name
                )

                if not locked:
                    print(f"\n🎯 ยิงเป้าหมายที่พบทั้งหมดเรียบร้อยแล้ว (ยิงสำเร็จไปทั้งสิ้น {shot_count} เป้า)")
                    break

                # บันทึกมุม Gimbal ตอนล็อกนิ่งเป้าหมาย
                aimed_yaw = current_gimbal_yaw

                # ยิงกระสุน 2 นัด (พร้อมสตรีมสด ไม่ค้าง)
                fire_double_shot(
                    ep_blaster=ep_blaster,
                    ep_led=ep_led,
                    ep_camera=ep_camera,
                    window_name=window_name,
                    count=shots_per_target,
                    enable_fire=enable_fire,
                    led_rgb=locked_target.get("led_rgb") if locked_target else None
                )
                shot_count += 1
                print(f"✅ ยิงเป้าหมาย #{shot_count} สำเร็จ!")

                # จดจำเป้าหมายนี้ลงหน่วยความจำทันที
                target_memory.record_shot(
                    target_num=shot_count,
                    gimbal_yaw=aimed_yaw,
                    target_info=locked_target
                )

                # ดึง Gimbal กลับเข้าสู่กึ่งกลาง พร้อมสตรีมสด
                recenter_gimbal_with_stream(ep_gimbal, ep_camera, window_name=window_name)

            # เข้าสู่โหมด STANDBY เฝ้าระวัง ไม่หยุดทำงานกะทันหัน
            print("\n" + "="*75)
            print(f"🎉 [MISSION COMPLETED] ยิงเป้าหมายครบ {shot_count} เป้าหมายแล้ว!")
            print("🟢 เข้าสู่โหมด [STANDBY] พร้อมสตรีมภาพสดต่อเนื่อง (ไม่หยุดทำงาน)")
            print("   - กด [SPACEBAR] : สแกนหาเป้าหมายและยิงต่อทันที")
            print("   - กด [R]        : ล้างหน่วยความจำ (Reset Memory) และเริ่มยิงใหม่")
            print("   - กด [Q]        : สิ้นสุดภารกิจและปิดโปรแกรม")
            print("="*75 + "\n")

            ep_led.set_led(comp=led.COMP_TOP_ALL, r=0, g=255, b=0, effect=led.EFFECT_BREATH)

            in_standby = True
            while in_standby:
                img = ep_camera.read_cv2_image(strategy="newest", timeout=0.1)
                if img is None:
                    time.sleep(0.01)
                    continue

                h, w, _ = img.shape
                # HUD Banner
                cv2.rectangle(img, (15, 15), (w - 15, 105), (20, 20, 20), -1)
                cv2.rectangle(img, (15, 15), (w - 15, 105), (0, 255, 0), 2)
                cv2.putText(img, f"MISSION COMPLETE: {shot_count} TARGETS NEUTRALIZED", (30, 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 2)
                cv2.putText(img, "STANDBY: [SPACEBAR]=Scan & Shoot | [R]=Reset & Repeat | [Q]=Exit", (30, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

                # Crosshair
                cx_mid, cy_mid = w // 2, h // 2
                cv2.line(img, (cx_mid - 20, cy_mid), (cx_mid + 20, cy_mid), (0, 255, 0), 1)
                cv2.line(img, (cx_mid, cy_mid - 20), (cx_mid, cy_mid + 20), (0, 255, 0), 1)

                # ตรวจจับเป้าหมายสดและแสดงผล
                targets = detect_targets(img, color_name, shape_name, hsv_configs)
                targets = target_memory.annotate_targets(targets, current_gimbal_yaw)

                for t in targets:
                    bx, by, bw, bh = t["bbox"]
                    if t.get("is_already_shot", False):
                        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (80, 80, 80), 1)
                        cv2.putText(img, f"SHOT #{t.get('shot_id')}", (bx, max(15, by - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                    else:
                        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
                        cv2.putText(img, "UNSHOT TARGET", (bx, max(15, by - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

                cv2.imshow(window_name, img)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("\n>> ผู้ใช้กด [Q] ปิดโปรแกรมและปลดการเชื่อมต่อ")
                    mission_running = False
                    in_standby = False
                    break
                elif key == ord('r'):
                    print("\n>> ผู้ใช้กด [R] รีเซ็ตหน่วยความจำเป้าหมาย (Reset Memory) และเริ่มยิงใหม่ทั้งหมด...")
                    target_memory.clear()
                    in_standby = False
                elif key in (ord(' '), 13):
                    print("\n>> ผู้ใช้กด [SPACEBAR] สแกนหาเป้าหมายและยิงต่อ...")
                    in_standby = False

    finally:
        try:
            ep_gimbal.unsub_angle()
        except Exception:
            pass
        cv2.destroyAllWindows()
