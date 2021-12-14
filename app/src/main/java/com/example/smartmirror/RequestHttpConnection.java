package com.example.smartmirror;

import android.content.ContentValues;
import android.util.Log;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

public class RequestHttpConnection {
    public static String postRequest(String _url, LinkedHashMap<String,String> _params) {

        HttpURLConnection urlConn = null;

        StringBuffer sbParams = new StringBuffer();

        if (_params == null)
            sbParams.append("");
        else {
            boolean isAnd = false;
            String key;
            String value;

            for(Map.Entry<String,String> parameter: _params.entrySet())
            {
                key=parameter.getKey();
                Log.e("contentkey",key);
                value=parameter.getValue().toString();
                Log.e("contentvalue",value);
                if(isAnd)
                {
                    sbParams.append("&");
                }
                sbParams.append(key).append("=").append(value);
                if(!isAnd)
                {
                    if(_params.size()>=2)
                    {
                        isAnd=true;
                    }
                }
            }
            Log.e("sbParams",String.valueOf(sbParams));

//            for(Map.Entry<String, Object> parameter : _params.valueSet()){
//                key = parameter.getKey();
//                Log.e("contentkey",key);
//                value = parameter.getValue().toString();
//                Log.e("contentvalue",value);
//                if (isAnd)
//                    sbParams.append("&");
//
//                sbParams.append(key).append("=").append(value);
//
//                if (!isAnd)
//                    if (_params.size() >= 2)
//                        isAnd = true;
//            }
        }

        try{
            URL url = new URL(_url);
            Log.e("url",String.valueOf(url));
            urlConn = (HttpURLConnection) url.openConnection();

            urlConn.setRequestMethod("POST");
            urlConn.setRequestProperty("Accept-Charset", "UTF-8");
            urlConn.setRequestProperty("Context_Type", "application/x-www-form-urlencoded;charset=UTF-8");

            String strParams = sbParams.toString();
            Log.e("strParams",strParams);
            OutputStream os = urlConn.getOutputStream();
            os.write(strParams.getBytes("UTF-8"));
            os.flush();
            os.close();

            if (urlConn.getResponseCode() != HttpURLConnection.HTTP_OK)
                return null;

            BufferedReader reader = new BufferedReader(new InputStreamReader(urlConn.getInputStream(), "UTF-8"));

            String line;
            String page = "";

            while ((line = reader.readLine()) != null){
                page += line;
            }

            return page;

        } catch (MalformedURLException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (urlConn != null)
                urlConn.disconnect();
        }

        return null;
    }

}

