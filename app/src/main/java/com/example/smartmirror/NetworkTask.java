package com.example.smartmirror;

import android.content.ContentValues;
import android.os.AsyncTask;

public class NetworkTask extends AsyncTask<Void,Void,String> {
    private String url;
    private ContentValues values;

    public NetworkTask(String url,ContentValues values){
        this.url=url;
        this.values=values;
    }

    @Override
    protected String doInBackground(Void... params) {
        String result;
        RequestHttpConnection requestHttpConnection=new RequestHttpConnection();

        return null;
    }
    protected void onPostExecute(String s)
    {
        super.onPostExecute(s);


    }
}
