import requests 
from bs4 import BeautifulSoup 
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import pandas as pd
import logging

file_handler =logging.FileHandler(
    filename='./abc.log',  # 로그 파일명 설정
    encoding='utf-8')

log=logging.getLogger()
log.addHandler(file_handler)
log.setLevel(logging.DEBUG)


def get_html(url):
    driver = webdriver.Chrome()
    
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    #센터명
    title=soup.find_all('span', class_='ant-page-header-heading-title')

    #도로명 주소, 평당 가격
    title_content=soup.find_all('div', class_='ant-descriptions ant-descriptions-small')

    #건축개요
    construct=soup.find_all('div', class_='WarehouseSummary__WarehouseSummaryWrapper-sc-1klkj2t-0 sCjVi')

    #접안시설
    facility=soup.find_all('div', class_='equipment')

    #건물정보
    building=soup.find_all('div', class_='WarehouseLedge__Wrapper-sc-1u38f37-0 elroIs')
    driver.quit()
    return title,title_content,construct, facility,building

def make_table(title,title_content,construct, facility, building):
    TITLES=[]
    VALUES=[]
    #센터명
    TITLES.append('센터 이름')
    VALUES.append(title[0].get('title'))


    title_cont=title_content[0].text.strip()
    title_ti=['도로명주소','지번주소','평당 가격']
    for i in range(len(title_ti)):
        if i==(len(title_ti)-1):
            vv=title_cont[title_cont.find(title_ti[i])+len(title_ti[i]):]
            VALUES.append(vv)
        else:
            vv=title_cont[title_cont.find(title_ti[i])+len(title_ti[i]):title_cont.find(title_ti[i+1])]
            VALUES.append(vv)
    
    TITLES=TITLES+title_ti  
   

    const_text=construct[0].text.strip()
    titles=['대지면적','연면적','건폐율','용적률','허가일자','착공일자','사용승인일자','규모','건물 내 주차','건물 외 주차']
    for i in range(len(titles)):
        if i==(len(titles)-1):
            vv=const_text[const_text.find(titles[i])+len(titles[i]):]
            VALUES.append(vv)
        else:
            vv=const_text[const_text.find(titles[i])+len(titles[i]):const_text.find(titles[i+1])]
            VALUES.append(vv)
    
    TITLES=TITLES+titles  
 

    facilty_text=facility[0].text.strip()
    facilty_titles=['접안시설','창고 설비','편의시설','인증 및 보험']
    for i in range(len(facilty_titles)):
        if i==(len(facilty_titles)-1):
            vv=facilty_text[facilty_text.find(facilty_titles[i])+len(facilty_titles[i]):]
            VALUES.append(vv)
        else:
            vv=facilty_text[facilty_text.find(facilty_titles[i])+len(facilty_titles[i]):facilty_text.find(facilty_titles[i+1])]
            VALUES.append(vv)
    
    TITLES=TITLES+facilty_titles  
  

    build_text=building[0].text.strip()
    build_titles=['건물(동) 정보','동명칭','연면적','건축면적','층수','가수동 근생및 자동차관련시설']
    for i in range(len(build_titles)):
        if i==(len(build_titles)-1):
            vv=build_text[build_text.find(build_titles[i])+len(build_titles[i]):]
            VALUES.append(vv)
        else:
            vv=build_text[build_text.find(build_titles[i])+len(build_titles[i]):build_text.find(build_titles[i+1])]
            VALUES.append(vv)
    
    TITLES=TITLES+build_titles  
 
    
    return TITLES, VALUES

Values=[]
for i in range(1,6890):
    url = "https://abc.co.kr/abc/{}".format(i) 
    print(url)
    try:
        title,title_content, construct, facility, building=get_html(url)
        TITLES, VALUES= make_table(title,title_content , construct, facility, building)
        Values.append(VALUES)
    except:
        continue

RESULT=pd.DataFrame(Values,columns=TITLES)
RESULT.to_csv('./abc.csv',encoding='cp949',index=False)