package com.example.smartmirror;

import android.app.ProgressDialog;
import android.graphics.drawable.Drawable;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ListView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;

import com.bumptech.glide.Glide;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;

public class ItemFragment2 extends Fragment {

    ListView itemList;
    ListItem listItem;
    UserItemAdapter adapter;
    private static String IP_ADDRESS="52.79.59.24";
    String TAG="php";
    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        return inflater.inflate(R.layout.fragment_item2, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        itemList = view.findViewById(R.id.itemList);
        listItem = new ListItem();
        adapter = new UserItemAdapter(getContext());
        Drawable pic1 = getResources().getDrawable(R.drawable.img6);
        Drawable pic2 = getResources().getDrawable(R.drawable.img7);
        Drawable pic3 = getResources().getDrawable(R.drawable.img8);
        Drawable pic4 = getResources().getDrawable(R.drawable.img9);
        Drawable pic5 = getResources().getDrawable(R.drawable.img10);

        adapter.addItem("", "");
        //Glide.with.load(listItem.getImage()).override(800, 800);
        //adapter.addItem(pic2,"https://www.youtube.com/watch?v=rod11dE9JaA");
        //adapter.addItem(pic3, "https://www.youtube.com/watch?v=Q-iEfzaikBI");
        //adapter.addItem(pic4, "https://www.youtube.com/watch?v=GA9GigGuf24");
        //adapter.addItem(pic5, "https://www.youtube.com/watch?v=y5paOQU66hg");

        itemList.setAdapter(adapter);

        Button button1 = view.findViewById(R.id.btn_top2);
        Button button2 = view.findViewById(R.id.btn_bottom2);
        Button button3 = view.findViewById(R.id.btn_outer2);
        Button button4 = view.findViewById(R.id.btn_etc2);


        //////////////////////////////////////////
        //////////////////////////////////////////

        //top
        button1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                adapter.category="top";
                GetUserItem task=new GetUserItem();
                task.execute("http://"+IP_ADDRESS+"/getUserItem.php","top");
            }
        });
        //bottom
        button2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                adapter.category="bottom";
                GetUserItem task=new GetUserItem();
                task.execute("http://"+IP_ADDRESS+"/getUserItem.php","bottom");
            }
        });
        //outer
        button3.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                adapter.category="outer";
                GetUserItem task=new GetUserItem();
                task.execute("http://"+IP_ADDRESS+"/getUserItem.php","outer");
            }
        });
        //etc - onepiece
        button4.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                adapter.category="onepiece";
                GetUserItem task=new GetUserItem();
                task.execute("http://"+IP_ADDRESS+"/getUserItem.php","onepiece");
            }
        });
    }

    class GetUserItem extends AsyncTask<String, Void,String> {

        ProgressDialog progressDialog;

        protected void onPreExecute()
        {
            super.onPreExecute();
            //progressDialog=ProgressDialog.show(CategoryFragment.this, "Please Wait",null,true,true);
        }
        protected void onPostExecute(String result) // doInBackground()로부터 리턴된 값이 onPostExecute()의 매개변수로 넘어옴.
        {
            super.onPostExecute(result);
            listItem = new ListItem();
            adapter = new UserItemAdapter(getContext());
            //adapter.styling=styling;

            try {
                Log.e(">>result: ",result);
                JSONObject jsonObject=new JSONObject(result); // 가장 큰 JSONObject
                JSONArray jsonArray=jsonObject.getJSONArray("User_items"); // 배열을 가져옴
                for(int i=0;i<jsonArray.length();i++)
                {
                    JSONObject object=jsonArray.getJSONObject(i);
                    String image_url=object.getString("Image").toString();
                    String id=object.getString("ID").toString();
                    adapter.addItem(image_url,id);

                }
                itemList.setAdapter(adapter);
            } catch (JSONException e) {
                e.printStackTrace();
            }

            Log.d(TAG, "POST response  - " + result);
        }
        @Override
        protected String doInBackground(String... params) {
            String category = (String)params[1];

            String serverURL = (String)params[0];
            //TODO: change serverURL
            serverURL = serverURL+"?" + "CATEGORY=" + category;
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
