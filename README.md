# **YOLO-DETECT**

**快速开始**

python版本

```
python==3.9
```

相关库版本

```python
numpy==2.2.4
opencv_contrib_python==4.10.0.84
opencv_python==4.10.0.84
PyQt5==5.15.11
PyQt5_sip==12.16.1
ultralytics==8.3.58
```

拉取代码

```
git clone https://github.com/crnono/cv_detect_qt.git
```

安装依赖

```
cd cv_detect_qt
conda create 环境
pip install -r requirements.txt
```

运行

```shell
conda activate 环境
cd burr_multivariate_recommend
python main.py
```

打包

```python
pip install pyinstaller
pyinstaller --onefile main.py
"""
参数说明：
--onefile：将所有依赖打包成一个单独的可执行文件。如果不加此参数，PyInstaller 会生成一个包含多个文件和文件夹的 dist 目录。
--windowed：如果是 GUI 程序（不需要命令行窗口），可以加上此参数，打包后的程序运行时不会弹出命令行窗口。
--icon=icon.ico：指定程序图标（需要准备好 icon.ico 文件）
"""
```



**贡献指南**

欢迎来到我们的项目！我们非常感谢您有兴趣参与贡献，并力求使整个过程公开透明且友好。以下是贡献的一般准则：

*代码贡献*

请先fork本项目的仓库并在单独的分支上进行开发工作。
遵守项目中已存在的编码风格和约定。
编写清晰、结构良好且带有注释的代码。
在您的拉取请求（Pull Request）中包含相关的单元测试

*问题报告与建议*

如果发现任何错误或功能缺陷，请通过issue系统详细描述问题，并提供复现步骤和相关环境信息。
提出功能改进建议或新特性需求时，请确保详尽说明并讨论其价值和可能实现的方式。

*文档更新*

欢迎提交关于项目文档的修订，包括但不限于用户手册、API文档以及教程示例。

*社区交流*

尊重他人，保持友善沟通，遵守公司的行为准则。
积极参与讨论，分享经验与见解，帮助解答其他贡献者的问题。
请您在开始贡献之前仔细阅读项目README文件或其他相关政策文档，了解具体的提交流程和规范要求。期待您的宝贵贡献！



**开发人员**

crnono@yeah.net 



**License**

The entire codebase is under [MIT license](LICENSE).