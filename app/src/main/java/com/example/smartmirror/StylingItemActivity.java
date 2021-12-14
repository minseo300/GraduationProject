package com.example.smartmirror;

import androidx.appcompat.app.AppCompatActivity;

import android.app.ProgressDialog;
import android.content.Intent;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.widget.ListView;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;

// 무신사 추천 스타일링의 개별 아이템을 보여줌

public class StylingItemActivity extends AppCompatActivity {

    ListView itemList;
    ListItem items;
    StylingItemAdapter adapter;


    String top_url;
    String outer_url;
    String bottom_url;
    String onepiece_url;


    String top;
    String outer;
    String bottom;
    String onepiece;

    String ID;
    String styling;
    private static String IP_ADDRESS="52.79.59.24";

    String TAG="php";
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_styling_item);

        Intent intent=getIntent();
        ID=intent.getStringExtra("ID");
        styling=intent.getStringExtra("styling");
        Log.e(">>>styling",styling);
        GetItem task=new GetItem();
        task.execute("http://"+IP_ADDRESS+"/getItem.php",ID,styling);
    }


    class GetItem extends AsyncTask<String, Void,String> {

        ProgressDialog progressDialog;

        protected void onPreExecute()
        {
            super.onPreExecute();

        }
        protected void onPostExecute(String result) // doInBackground()로부터 리턴된 값이 onPostExecute()의 매개변수로 넘어옴.
        {
            super.onPostExecute(result);
            items = new ListItem();
            adapter = new StylingItemAdapter(getApplicationContext());
            itemList = findViewById(R.id.itemList);

            try {
                Log.e(">>result: ",result);
                JSONObject jsonObject=new JSONObject(result); // 가장 큰 JSONObject
                JSONArray jsonArray=jsonObject.getJSONArray("Item"); // 배열을 가져옴
                for(int i=0;i<jsonArray.length();i++)
                {

                    JSONObject object=jsonArray.getJSONObject(i);
                    String category=object.getString("Category").toString();
                    Log.e("category: ",category);

                   if(category.equals("top"))
                    {
                        String image_url=object.getString("Image").toString();
                        Log.e("image-top:",image_url);
                        String top_link=object.get("topLink").toString();
                        // String link=object.get("topLink").toString();
                        adapter.addItem(image_url,top_link);
                    }
                    else if(category.equals("bottom"))
                    {
                        String image_url=object.getString("Image").toString();
                        Log.e("image-bottom:",image_url);
                        String bottom_link=object.get("bottomLink").toString();
                        //String link=object.get("bottomLink").toString();
                        adapter.addItem(image_url,bottom_link);
                    }
                    else if(category.equals("outer"))
                    {
                        String image_url=object.getString("Image").toString();
                        Log.e("image-outer:",image_url);
                        String outer_link=object.getString("outerLink").toString();
                        //String link=object.get("outerLink").toString();
                        adapter.addItem(image_url,outer_link);
                    }
                    else if(category.equals("onepiece"))
                    {
                        String image_url=object.getString("Image").toString();
                        Log.e("image-onepiece:",image_url);
                        String onepiece_link=object.getString("onepieceLink").toString();
                        //String link=object.get("onepieceLink").toString();
                        adapter.addItem(image_url,onepiece_link);
                    }
                    //adapter.addItem(image_url,link);
                }
                itemList.setAdapter(adapter);
            } catch (JSONException e) {
                e.printStackTrace();
            }
            Log.d(TAG, "POST response  - " + result);
        }
        @Override
        protected String doInBackground(String... params) {
            String temp = (String)params[1];
            String styling = (String)params[2];

            String serverURL = (String)params[0];
            serverURL = serverURL+"?" + "ID=" + temp + "&STYLING=" + styling ;
            Log.d("url: ",String.valueOf(serverURL));
            try {
                URL url = new URL(serverURL);
                HttpURLConnection httpURLConnection = (HttpURLConnection) url.openConnection();

                httpURLConnection.setReadTimeout(5000);
                httpURLConnection.setConnectTimeout(5000);
                httpURLConnection.setRequestMethod("GET");
                httpURLConnection.connect();

                int responseStatusCode = httpURLConnection.getResponseCode();
                Log.d(TAG, "GET response code - " + responseStatusCode);

                InputStream inputStream;
                if(responseStatusCode == HttpURLConnection.HTTP_OK) {
                    inputStream = httpURLConnection.getInputStream();
                }
                else{
                    inputStream = httpURLConnection.getErrorStream();
                }
                InputStreamReader inputStreamReader = new InputStreamReader(inputStream, "UTF-8");
                BufferedReader bufferedReader = new BufferedReader(inputStreamReader);

                StringBuilder sb = new StringBuilder();
                String line = null;

                while((line = bufferedReader.readLine()) != null){
                    sb.append(line);
                }
                bufferedReader.close();
                Log.e("Result - Sb: ",String.valueOf(sb));

                return sb.toString();

            } catch (Exception e) {
                Log.d(TAG, "GetData: Error ", e);
                return new String("Error: " + e.getMessage());
            }
        }
    }
}