import pandas as pd
import pymysql

db=pymysql.connect(host="54.180.67.155", user="minseo",password="minseopw", db="SmartMirror",charset='utf8')
curs=db.cursor()

topCate_bottomCate=pd.read_pickle("topCate_bottomCate.pkl")
topColor_bottomColor=pd.read_pickle("topColor_bottomColor.pkl")
topCate_bottomFit=pd.read_pickle("topCate_bottomFit.pkl")
topCate_outerCate=pd.read_pickle("topCate_outerCate.pkl")
topColor_outerColor=pd.read_pickle("topColor_outerColor.pkl")
outerColor_bottomColor=pd.read_pickle("outerColor_bottomColor.pkl")
outerCate_bottomCate=pd.read_pickle("outerCate_bottomCate.pkl")
outerCate_bottomFit=pd.read_pickle("outerCate_bottomFit.pkl")
temperature_section_file=open('temperature_section.txt','r')
temperature_section=int(temperature_section_file.read())

if temperature_section>=3 and temperature_section<=7:
    top_temperature_section=3
    bottom_temperature_section=3
else:
    top_temperature_section=temperature_section
if temperature_section==1 or temperature_section==2:
    bottom_temperature_section=1



# for top and bottom
def top_bottom(top_temperature_section,bottom_temperature_section):
    sql="SELECT  t.ID, b.ID, t.category,b.category, t.color, b.color, b.bottom_fit FROM U_top_info as t JOIN U_bottom_info as b WHERE t.temperature_section="+str(top_temperature_section)+" AND b.temperature_section="+str(bottom_temperature_section)+";"
    curs.execute(sql)
    result=pd.DataFrame(curs.fetchall())
    print(result)

    result_size=len(result) # 모든 경우의 수 갯수
    scores=[]
    for index,row in result.iterrows():
        topID=row[0]
        bottomID=row[1]
        topCate=row[2]
        bottomCate=row[3]
        topColor=row[4]
        bottomColor=row[5]
        bottomFit=row[6]
        score=0
        combi=[topID,bottomID]

        score=topCate_bottomCate.loc[bottomCate][topCate]+topColor_bottomColor.loc[bottomColor][topColor]+topCate_bottomFit.loc[bottomFit][topCate]
        scores.append([combi,score])

# for outer, top, bottom
def outer_top_bottom(temperature_section,top_temperature_section,bottom_temperature_section):
    sql="SELECT  t.ID, b.ID, t.category,b.category, t.color, b.color, b.bottom_fit, o.ID,o.category,o.color FROM U_top_info as t JOIN U_bottom_info as b JOIN U_outer_info as o WHERE t.temperature_section="+str(top_temperature_section)+" AND b.temperature_section="+str(bottom_temperature_section)+" AND o.temperature_section="+str(temperature_section)+";"
    curs.execute(sql)
    result=pd.DataFrame(curs.fetchall())
    print(result)

    result_size=len(result) # 모든 경우의 수 갯수
    scores=[]
    for index,row in result.iterrows():
        topID=row[0]
        bottomID=row[1]
        topCate=row[2]
        bottomCate=row[3]
        topColor=row[4]
        bottomColor=row[5]
        bottomFit=row[6]
        outerID=row[7]
        outerCate=row[8]
        outerColor=row[9]
        score=0
        combi=[topID,bottomID,outerID]
        print("1.",topCate_bottomCate.loc[bottomCate][topCate])
        print("2.",topColor_bottomColor.loc[bottomColor][topColor])
        print("3.",topCate_bottomFit.loc[bottomFit][topCate])
        print("4.",topCate_outerCate.loc[outerCate][topCate])
        print("5.",topColor_outerColor.loc[outerColor][topColor])
        print("6.",outerColor_bottomColor.loc[bottomColor][outerColor])
        print("7.",outerCate_bottomCate.loc[bottomCate][outerCate])
        print("8.",outerCate_bottomFit.loc[bottomFit][outerCate])

        topCate_bottomCate_score=topCate_bottomCate.loc[bottomCate][topCate]
        topColor_bottomColor_score = topColor_bottomColor.loc[bottomColor][topColor]
        if bottomFit=="null":
            topCate_bottomFit_score=0
        else:
            topCate_bottomFit_score = topCate_bottomFit.loc[bottomFit][topCate]
        topCate_outerCate_score = topCate_outerCate.loc[outerCate][topCate]
        topColor_outerColor_score = topColor_outerColor.loc[outerColor][topColor]
        outerColor_bottomColor_score = outerColor_bottomColor.loc[bottomColor][outerColor]
        outerCate_bottomCate_score = outerCate_bottomCate.loc[bottomCate][outerCate]
        if bottomFit == "null":
            outerCate_bottomFit_score=0
        else:
            outerCate_bottomFit_score = outerCate_bottomFit.loc[bottomFit][outerCate]


        score=topCate_bottomCate_score+topColor_bottomColor_score+topCate_bottomFit_score+topCate_outerCate_score+topColor_outerColor_score+outerColor_bottomColor_score+outerCate_bottomCate_score+outerCate_bottomFit_score
        # score=topCate_bottomCate.loc[bottomCate][topCate]+topColor_bottomColor.loc[bottomColor][topColor]+topCate_bottomFit.loc[bottomFit][topCate]+\
        #       topCate_outerCate.loc[outerCate][topCate]+topColor_outerColor.loc[outerColor][topColor]+outerColor_bottomColor.loc[bottomColor][outerColor]+outerCate_bottomCate.loc[bottomCate][outerCate]+outerCate_bottomFit.loc[bottomFit][outerCate]
        scores.append([combi,score])

if temperature_section<4:
    top_bottom(top_temperature_section,bottom_temperature_section)
else:
    outer_top_bottom(temperature_section,top_temperature_section,bottom_temperature_section)