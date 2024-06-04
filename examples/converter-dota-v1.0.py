from ultralytics.data.converter import convert_dota_to_yolo_obb

"""
    将DOTA数据转成yolo-obb格式
"""

DOTA_PATH = "N:/dataset/DOTA/DOTA-v1.0-yolov8-obb"

convert_dota_to_yolo_obb(DOTA_PATH)