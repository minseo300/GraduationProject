import pymysql
db=pymysql.connect(host="54.180.67.155", user="minseo",password="minseopw", db="SmartMirror",charset='utf8')
curs=db.cursor()

#get category`s id from database
def get_id(category):
    sql="SELECT "+category+"_ from user_clothes_id_info;"
    curs.execute(sql)
    result=curs.fetchone()
    return result[0]

#set category`s id in database
def set_id(category,id):
    sql = "UPDATE user_clothes_id_info SET " + category + "_=" + str(id) + ";"
    curs.execute(sql)
    db.commit()
