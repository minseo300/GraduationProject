package com.example.smartmirror;

import android.app.Activity;
import android.graphics.drawable.Drawable;
import android.widget.ImageButton;

import com.bumptech.glide.Glide;
// 스타일링
public class ListItem {

    private String url; //이미지
    private String link;
    private String ID;
    private String category; // for user item; 사용자 옷 뭐 있는지 앱에 보여줄 때 사용

    // 사용자 옷으로 추천해줄 때 사용
    String outer;
    String top;
    String bottom;
    String outerID;
    String topID;
    String bottomID;

    public void setOuter(String outer){this.outer=outer;} //이미지 url
    public void setTop(String top){this.top=top;}
    public void setBottom(String bottom){this.bottom=bottom;}
    public String getOuter(){return outer;}
    public String getTop(){return top;}
    public String getBottom(){return bottom;}

    public void setOuterID(String outerID){this.outerID=outerID;}
    public void setTopID(String topID){this.topID=topID;}
    public void setBottomID(String bottomID){this.bottomID=bottomID;}
    public String getOuterID(){return outerID;}
    public String getTopID(){return topID;}
    public String getBottomID(){return bottomID;}

    ///////////////////////////////////////////


    public void setImage(String url)
    {
      this.url=url;
    } //s3에 업로드 되어있는 주소
    public void setLink(String link) {
        this.link = link ;
    } // 무신사 옷 아이템 판매 링크
    public void setID(String ID){this.ID=ID;} // 데이터베이스에 이미지들이 id(primary key *R__image에서는 id가 primary key가 아님)로 저장되어있음
    public void setCategory(String category){ this.category=category;} // for user item; 사용자 옷 뭐 있는지 앱에 보여줄 때 사용 - top,bottom,outer,onepiece

    public String getImage() {
        return this.url;
    }
    public String getLink() {
        return this.link;
    }
    public String getID() {
        return this.ID;
    }

}
