# Collacte Human descriptions from indoor view images

## 1.Generate human motion prompt text with gpt4
制作建筑中所有视点的可视化坐标图
输入：scan_id 建筑扫描编号
读取视点坐标文件con/pos_info/{scan_id}_pos_info.json
文件结构{key:视点编号 viewpoint_id, value:坐标[x,y,z]}，如{"10c252c90fa24ef3b698c6f54d984c5c": [-5.48891,1.4484,1.53509]}
要求：1、显示网格，单位网格边长为1;2、鼠标左键点击视点，下方文本框输出视点编号
语言：python
## 2.Placing human at the viewpoints (manual)