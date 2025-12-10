import cv2
import dlib


def add_christmas_hat_dlib(face_img_path, hat_img_path, output_path="person_with_hat.jpg"):
    """
    基于Dlib的精准人脸检测和圣诞帽叠加
    :param face_img_path: 人脸图片路径
    :param hat_img_path: 圣诞帽PNG（透明底）路径
    :param output_path: 结果保存路径
    :return: None
    """
    # 1. 加载Dlib模型和图片
    detector = dlib.get_frontal_face_detector()  # 人脸检测器
    predictor = dlib.shape_predictor("./resources/shape_predictor_68_face_landmarks.dat")  # 关键点模型

    img = cv2.imread(face_img_path)
    if img is None:
        raise ValueError(f"无法读取人脸图片：{face_img_path}")

    hat_img = cv2.imread(hat_img_path, cv2.IMREAD_UNCHANGED)  # 带透明通道的圣诞帽
    if hat_img is None:
        raise ValueError(f"无法读取圣诞帽图片：{hat_img_path}")

    # 2. Dlib人脸检测（支持多尺度检测，精度更高）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 使用1表示缩放因子，增加检测精度；upsample_num_times=1表示对图片进行一次上采样，检测更小的人脸
    faces = detector(gray, 1)

    if len(faces) == 0:
        print("警告：未检测到人脸，直接保存原图")
        cv2.imwrite(output_path, img)
        return

    print(f"Dlib检测到 {len(faces)} 个人脸")

    # 3. 为每个人脸叠加圣诞帽
    for face in faces:
        # 3.1 获取68个面部关键点
        landmarks = predictor(gray, face)

        # 3.2 精确定位头顶位置（使用额头关键点19和24）
        forehead_left = landmarks.part(19)  # 左额头关键点
        forehead_right = landmarks.part(24)  # 右额头关键点
        hat_center_x = (forehead_left.x + forehead_right.x) // 2  # 帽子水平居中
        hat_top_y = min(forehead_left.y, forehead_right.y)  # 头顶位置（取额头最高点）

        # 3.3 基于人脸宽度自适应调整帽子大小
        face_width = face.right() - face.left()
        hat_target_width = int(face_width * 1.5)  # 帽子宽度为脸宽的1.5倍
        hat_h, hat_w = hat_img.shape[:2]
        scale = hat_target_width / hat_w
        hat_target_height = int(hat_h * scale)

        # 3.4 等比例缩放帽子
        hat_resized = cv2.resize(hat_img, (hat_target_width, hat_target_height))

        # 3.5 计算帽子叠加位置
        hat_x = hat_center_x - hat_target_width // 2
        hat_y = hat_top_y - hat_target_height

        # 3.6 处理边界问题，确保帽子在图片范围内
        y1, y2 = max(0, hat_y), min(img.shape[0], hat_y + hat_target_height)
        x1, x2 = max(0, hat_x), min(img.shape[1], hat_x + hat_target_width)
        hat_y1, hat_y2 = y1 - hat_y, y2 - hat_y
        hat_x1, hat_x2 = x1 - hat_x, x2 - hat_x

        # 3.7 透明通道融合（精准处理，避免遮挡人脸）
        if y1 < y2 and x1 < x2:
            alpha = hat_resized[hat_y1:hat_y2, hat_x1:hat_x2, 3] / 255.0
            for c in range(3):
                img[y1:y2, x1:x2, c] = (1 - alpha) * img[y1:y2, x1:x2, c] + alpha * hat_resized[
                    hat_y1:hat_y2, hat_x1:hat_x2, c]

    # 4. 保存结果
    cv2.imwrite(output_path, img)
    print(f"结果已保存到 {output_path}")


if __name__ == "__main__":
    # 人脸图片路径（JPG/PNG均可）
    FACE_IMG = "C:\\Users\\Administrator\\Desktop\\W020180330321867349838.jpg"
    # 圣诞帽PNG（透明底）路径
    HAT_IMG = "./resources/christmas_hat.png"
    # 结果保存路径（自动根据输入文件名添加 "_with_hat" 后缀）
    OUTPUT_IMG = FACE_IMG.replace(".jpg", "_with_hat.jpg").replace(".png", "_with_hat.png")

    try:
        add_christmas_hat_dlib(FACE_IMG, HAT_IMG, OUTPUT_IMG)
    except Exception as e:
        print(f"运行出错：{e}")
