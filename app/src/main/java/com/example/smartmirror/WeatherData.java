package com.example.smartmirror;

import android.util.Log;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLEncoder;
import java.text.SimpleDateFormat;
import java.util.Date;

public class WeatherData {
    //int weather;
    String nowtime;
    String date;
    double weather=0.0;

    public void getWeather(String x, String y) throws IOException, JSONException {

        String full_date_time=getDateTime();
        //Log.e("full_date_time: ",full_date_time);
        date=full_date_time.substring(0,8);
        Log.e("today: ",date);
        nowtime=full_date_time.substring(8,10)+"00";
        //nowtime="0800";
        Log.e("now time",nowtime);
        String time=timeChange(nowtime);
        Log.e("time: ",time);
        String apiUrl = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"; // http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst //getUltraSrtFcst
        String serviceKey = "HWWcc7ydy6e5YT2AQb5sVwkGOATdTjIQhPRVfqXoNTJsTTfBBpDBHpY23to4K1dC1fwEo5rcBq3coBKL%2B7qidQ%3D%3D";
        String nx = x;
        String ny = y;
        //Log.e("nx: ",nx);
        String baseDate = date;
        String baseTime = time;
        String type = "JSON";
        String numOfRows="100"; //306 217

        StringBuilder urlBuilder = new StringBuilder(apiUrl);
        try {
            urlBuilder.append("?" + URLEncoder.encode("ServiceKey", "UTF-8") + "=" + serviceKey);
            urlBuilder.append("&" + URLEncoder.encode("nx", "UTF-8") + "=" + URLEncoder.encode(nx, "UTF-8")); //경도
            urlBuilder.append("&" + URLEncoder.encode("ny", "UTF-8") + "=" + URLEncoder.encode(ny, "UTF-8")); //위도
            urlBuilder.append("&" + URLEncoder.encode("base_date", "UTF-8") + "=" + URLEncoder.encode(baseDate, "UTF-8")); /* 조회하고싶은 날짜*/
            urlBuilder.append("&" + URLEncoder.encode("base_time", "UTF-8") + "=" + URLEncoder.encode(baseTime, "UTF-8")); /* 조회하고싶은 시간 AM 02시부터 3시간 단위 */
            urlBuilder.append("&" + URLEncoder.encode("dataType", "UTF-8") + "=" + URLEncoder.encode(type, "UTF-8"));    /* 타입 */
            urlBuilder.append("&" + URLEncoder.encode("numOfRows", "UTF-8") + "=" + URLEncoder.encode(numOfRows, "UTF-8"));    /* 한 페이지 결과 수 */
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        }
        String url = urlBuilder.toString();
        RequestThread requestThread=new RequestThread(url);
        requestThread.start();

    }
    public String getDateTime(){
        long now=System.currentTimeMillis();
        Date date=new Date(now);
        SimpleDateFormat format=new SimpleDateFormat("yyyyMMddHHmmss");

        return format.format(date);
    }
    public String timeChange(String time)
    {
        // 현재 시간에 따라 데이터 시간 설정(3시간 마다 업데이트) //
        /**
         시간은 3시간 단위로 조회해야 한다. 안그러면 정보가 없다고 뜬다.
         0200, 0500, 0800 ~ 2300까지
         그래서 시간을 입력했을때 switch문으로 조회 가능한 시간대로 변경해주었다.
         **/
        switch(time) {

            case "0200":
            case "0300":
            case "0400":
            case "0500":
                time = "0200";
                break;

            case "0600":
            case "0700":
            case "0800":
                time = "0500";
                break;

            case "0900":
            case "1000":
            case "1100":
                time = "0800";
                break;

            case "1200":
            case "1300":
            case "1400":
                time = "1100";
                break;

            case "1500":
            case "1600":
            case "1700":
                time = "1400";
                break;

            case "1800":
            case "1900":
            case "2000":
                time = "1700";
                break;

            case "2100":
            case "2200":
                time = "2000";
                break;
            case "2300":
            case "0000":
            case "0100":
                time = "2300";
                date=String.valueOf(Integer.parseInt(date)-1);
                Log.e("yesterday",date);
                break;

        }
        return time;
    }
    class RequestThread extends Thread{
        String _url;
        String result="";

        RequestThread(String _url)
        {
            this._url=_url;
        }
        public void run()
        {
            try
            {
                //Temperature temperature=new Temperature();
                /*
                 * GET방식으로 전송해서 파라미터 받아오기
                 */
                Log.e("Here-","1");
                URL url = new URL(_url);
                Log.d("url: ",String.valueOf(url));
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("GET");
                conn.setRequestProperty("Content-type", "application/json");
                System.out.println("Response code: " + conn.getResponseCode());
                Log.e("response code: ",String.valueOf(conn.getResponseCode()));
                BufferedReader rd;
                if (conn.getResponseCode() >= 200 && conn.getResponseCode() <= 300) {
                    rd = new BufferedReader(new InputStreamReader(conn.getInputStream()));
                } else {
                    rd = new BufferedReader(new InputStreamReader(conn.getErrorStream()));
                }

                StringBuilder sb = new StringBuilder();
                String line;
                while ((line = rd.readLine()) != null) {
                    Log.e("Line: ",line);
                    sb.append(line);
                }

                rd.close();
                conn.disconnect();
                result = sb.toString();
                Log.d(">>Result: ",result);
                JSONObject jsonObj_1 = new JSONObject(result);
                String response = jsonObj_1.getString("response");
                Log.e("response: ",response);

                // response 로 부터 body 찾기
                JSONObject jsonObj_2 = new JSONObject(response);
                String body = jsonObj_2.getString("body");

                // body 로 부터 items 찾기
                JSONObject jsonObj_3 = new JSONObject(body);
                String items = jsonObj_3.getString("items");
                Log.i("ITEMS", items);

                // items로 부터 itemlist 를 받기
                JSONObject jsonObj_4 = new JSONObject(items);
                JSONArray jsonArray = jsonObj_4.getJSONArray("item");

                for (int i = 0; i < jsonArray.length(); i++)
                {
                    jsonObj_4 = jsonArray.getJSONObject(i);
                    String fcstValue = jsonObj_4.getString("fcstValue");
                    //Log.e("fcstValue: ",fcstValue);
                    String category = jsonObj_4.getString("category");
                    //Log.e("category: ",category);
                    String fcstTime=jsonObj_4.getString("fcstTime");
                    //Log.e("fcstTime: ",fcstTime);
                    //Log.e("Now time: ",nowtime);
                    String fcstDate=jsonObj_4.getString("fcstDate");
                    //Log.e("fcstDate: ",fcstDate);
                    if(category.equals("TMP")&&fcstTime.equals(nowtime))
                    {
                        weather=Double.parseDouble(fcstValue);
                        Log.e("TMP___: ",String.valueOf(weather));
                    }

                }

            } catch(Exception e) {
                e.printStackTrace();
            }
        }

    }

}
