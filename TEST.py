# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

# 폰트 설정 (Windows 기준: 맑은 고딕)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 음수 깨짐 방지

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title("자비스의 첫 플롯!")
plt.xlabel("X 축")
plt.ylabel("Y 축")
plt.grid(True)
plt.show()