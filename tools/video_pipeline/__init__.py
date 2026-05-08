"""動画解析 pipeline (Session #48 C、 dev/video-poc).

5 module:
- download.py: 動画 download (netkeiba 調教 / JRA-VAN ネクスト)
- yolo_inference.py: YOLOv8 馬体検出
- keypoint_extract.py: 歩様 keypoint (DLC SuperAnimal、 zero-shot)
- features_aggregate.py: features 化 (stride / pose / 体格)
- main_pipeline.py: 統合
"""
