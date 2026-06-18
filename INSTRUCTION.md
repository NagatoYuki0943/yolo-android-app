# 说明

这个项目打算做一个安卓可用的 yolo 部署 app。

yolo python 环境为 `conda activate yolo`，python 操作可以在这个环境中，如果没有包请自行安装。

yolo 路径在 `C:\ml\code\dl\yolo`

yolo 模型权重存放位置 `C:\ml\code\dl\yolo\weights`

已经导出了 `yolo26.onnx` ，`yolo26n_ncnn_model` 和 `yolo11.onnx`，图片输入大小都为 `[1, 3, 640, 640]`，输出则不同，yolo26.onnx 输出为 `[1, 300, 6]`，含义为 600 框和类别、分数、框的信息；`yolo26n_ncnn_model` 和 `yolo11.onnx` 的输出则为 `[1, 84, 8400]`，84 代表坐标信息和80个类别的得分，8400 代表8400 个框。

我想使用 ncnn 和 onnxruntime 实现在安卓的 GPU、CPU 上运行 yolo 模型。

powershell 代理 `$env:hTTP_PROXY="http://127.0.0.1:7890"; $env:HTTPS_PROXY="http://127.0.0.1:7890"`

需要手机调试可以通知我，我来负责打开软件、截屏等操作。
