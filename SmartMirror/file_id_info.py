import pymysql


db=pymysql.connect(host="52.79.59.24", user="minseo",password="minseopw", db="SmartMirror",charset='utf8')
curs=db.cursor()



def get_id(category):
    sql="SELECT "+category+"_ from user_clothes_id_info;"
    curs.execute(sql)
    result=curs.fetchone()
    # print(result[0])

    return result[0]
    # file=open('file_id_info.txt','r')
    # line=file.readline()
    # split=line.split(':')
    # outer_id=split[1]
    # outer_id=file.readline().split(':')[1]
    # top_id=file.readline().split(':')[1]
    # bottom_id=file.readline().split(':')[1]
    # if category=="outer":
    #     return outer_id
    # elif category=="top":
    #     return top_id
    # elif category=="bottom":
    #     return bottom_id
get_id("top")

def set_id(category,id):
    sql="UPDATE user_clothes_id_info SET "+category+"_="+id+";"
    curs.execute(sql)
    db.commit
