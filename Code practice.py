# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 14:09:53 2025

@author: gusrb
"""

# 1. 통합된 Excel Data 대상으로 계산 (by DH ksp DATA 처리코드)
# 2. 파장입력값으로 line ratio 출력

import pandas as pd # 표, 데이터를 사용하기 위한 모듈선언, pd로 별명
import glob # 파일 경로 및 불러오기 위한 모듈선언, glow 별명 없음
import numpy as np # 계산을 위한 모듈선언 (평균 및 line ratio)

files = glob.glob("C:/Users/gusrb/Desktop/WGS 실험 DATA/20250704 분석data/*.csv") #엑셀데이터 불러오기
file_number = len(files) #파일갯수 저장

print(f"검색된 file {file_number}개가 검색되었습니다") #검색된 파일갯수 확인

df = pd.read_csv(files[0],header=None) #첫번째파일 데이터프레임화
time_row = df.iloc[0,0] #데이터프레임중 time항목 추출 및 저장
wave_length = df.iloc[0,1:].astype(float) #데이터프레임중 wavelength항목 추출 및 저장

#print(f"time항목은 = {time_row}")
#print(f"wave항목은 = {wave_length}")

for file_name in file_number:  #file name 저장
    file_name = file_name.split('/')[-1]
    file_name.append(file_name)




#for i in range(file_number):
#    data=df.iloc[1:,0] #데이터프레임 저장


"""
for each_file in files:  #time과 파장을 저장,for문 돌릴 필요 없을것같은데
# 각 파일의 첫번째 행들을 for문안에서 리스트로 저장
# each_files 변수 / files 반복할 대상

    #***파일명 저장***
    file_name=each_file.split('/')[-1] #파일명 저장
    print(f"file name 확인 = {file_name}") #파일명 정상저장 확인

    #***엑셀파일의 첫행 리스트로 변환 및 저장***
    df=pd.read_csv(each_file, header=None) # 현재 파일의 첫번째 행을 읽어오기, header 설정해주기
    # 데이터프레임중 어디를 리스트로 가져올것이냐
    # 데이터프레임을 다시 리스트화 시켜줘야한다 df.iloc[행,열].tolist()
    
    time_row=df.iloc[0,0]    #dataframe의 time 항목 저장
    wavelegth_row=df.iloc[0,1:].astype(float) #dataframe의 wavelegth 항목 저장
    
    print(f"분류된 time cell = {time_row}")
    print(f"분류된 wavelength={wavelegth_row}")

    data=df.iloc[1:,1:].astype(float) #데이터프레임의 intensity값을 저장, 리스트는 아님
    
"""
    
# for i in file_number:
    
    
    




