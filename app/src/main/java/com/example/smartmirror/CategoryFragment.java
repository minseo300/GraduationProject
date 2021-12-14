package com.example.smartmirror;

import android.annotation.SuppressLint;
import android.app.ProgressDialog;
import android.content.Intent;
import android.graphics.drawable.Drawable;
import android.os.AsyncTask;
import android.os.Bundle;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;

import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ListView;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;


public class CategoryFragment extends Fragment {

    ListView categoryList;
    ListItem listItem;
    ListAdapter adapter;
    private static String IP_ADDRESS="52.79.59.24";
    String temp;
    String TAG="php";
    StylingItemAdapter s_adapter;
    String styling;

    @SuppressLint("LongLogTag")
    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        Bundle bundle=this.getArguments();
        temp=bundle.getString("temp_section");
        //Log.e(">>3.Temperature section: ",temp);
        return inflater.inflate(R.layout.fragment_category, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        categoryList = view.findViewById(R.id.categoryList);
        listItem = new ListItem();
        adapter = new ListAdapter(getContext());


        categoryList.setAdapter(adapter);

        Button button1 = view.findViewById(R.id.btn_dandy);
        Button button2 = view.findViewById(R.id.btn_formal);
        Button button3 = view.findViewById(R.id.btn_casual);
        Button button4 = view.findViewById(R.id.btn_street);
        Button button5 = view.findViewById(R.id.btn_chic);
        Button button6 = view.findViewById(R.id.btn_sport);
        Button button7 = view.findViewById(R.id.btn_romantic);
        Button button8 = view.findViewById(R.id.btn_girlish);

        //////////////////////////////////////////
        //////////////////////////////////////////

        // dandy
        button1.setOnClickListener(new View.OnClickListener() {
            @SuppressLint("LongLogTag")
            @Override
            public void onClick(View view) {

                Log.e(">>4. Temperature section: ",temp);
                styling="dandy";
                Log.e("button-styling",styling);
                adapter.styling=styling;
                //adapter.setStyling(styling);
                GetData task=new GetData();
                task.execute("http://"+IP_ADDRESS+"/getData2.php","6",styling);



            }
        });
        // formal
        button2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                Log.e(">>4. Temperature: ",temp);
                styling="formal";
                Log.e("button-styling",styling);
                adapter.styling=styling;
                GetData task=new GetData();
                task.execute("http://"+IP_ADDRESS+"/getData2.php","2",styling);
            }
        });
        // casual
        button3.setOnClickListener(new View.OnClickListener() {
            @SuppressLint("LongLogTag")
            @Override
            public void onClick(View view) {
                Log.e(">>4. Temperature section: ",temp);
                styling="casual";
                Log.e("button-styling",styling);
                adapter.styling=styling;
                GetData task=new GetData();
                task.execute("http://"+IP_ADDRESS+"/getData2.php","7",styling);




            }
        });
        // street
        button4.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                styling="street";
                Log.e("button-styling",styling);
                adapter.styling=styling;
                GetData task=new GetData();
                task.execute("http://"+IP_ADDRESS+"/getData2.php","5",styling);
            }
        });
        // chic
        button5.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Log.e(">>4. Temperature: ",temp);
                styling="chic";
                Log.e("button-styling",styling);
                adapter.styling=styling;
                GetData task=new GetData();
                task.execute("http://"+IP_ADDRESS+"/getData2.php","5",styling);
            }
        });
        // sport
        button6.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Log.e(">>4. Temperature: ",temp);
                styling="sports";
                Log.e("button-styling",styling);
                adapter.styling=styling;
                GetData task=new GetData();
                task.execute("http://"+IP_ADDRESS+"/getData2.php","5",styling);
            }
        });
        // romantic
        button7.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Log.e(">>4. Temperature: ",temp);
                styling="romantic";
                Log.e("button-styling",styling);
                adapter.styling=styling;
                GetData task=new GetData();
                task.execute("http://"+IP_ADDRESS+"/getData2.php","5",styling);
            }
        });
        // girlish
        button8.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Log.e(">>4. Temperature: ",temp);
                styling="girlish";
                Log.e("button-styling",styling);
                adapter.styling=styling;
                GetData task=new GetData();
                task.execute("http://"+IP_ADDRESS+"/getData2.php","5",styling);

            }
        });


    }
    class GetData extends AsyncTask<String, Void,String> {

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
            adapter = new ListAdapter(getContext());
            adapter.styling=styling;

            try {
                Log.e(">>result: ",result);
                JSONObject jsonObject=new JSONObject(result); // 가장 큰 JSONObject
                JSONArray jsonArray=jsonObject.getJSONArray("Look"); // 배열을 가져옴
                for(int i=0;i<jsonArray.length();i++)
                {
                    JSONObject object=jsonArray.getJSONObject(i);
                    String category=object.getString("Category").toString();
                    Log.e("category: ",category);

                    if(category.equals("styling")) {
                        String id = object.getString("ID").toString();
                        Log.e("frag-id: ", id);
                        listItem.setID(id);
                        Log.e("listitem id", listItem.getID());
                        String image_url = object.getString("Image").toString();
                        Log.e("image_url: ", image_url);
                        String temperature_section = object.getString("Temp").toString();
                        Log.e("temperature_section: ", temperature_section);
                        String outer_link = object.getString("outerLink").toString();
                        //Log.e("outer_link: ",outer_link);
                        String top_link = object.get("topLink").toString();
                        //Log.e("top_link: ",top_link);
                        String bottom_link = object.get("bottomLink").toString();
                        //Log.e("bottom_link: ",bottom_link);

                        //adapter.addItem(image_url, "https://www.youtube.com/watch?v=InZ_XAs0-nM",id);
                        adapter.addItem(image_url, "", id);
                    }
                }
                categoryList.setAdapter(adapter);
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
            serverURL = serverURL+"?" + "TEMP=" + temp + "&STYLING=" + styling ;
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
