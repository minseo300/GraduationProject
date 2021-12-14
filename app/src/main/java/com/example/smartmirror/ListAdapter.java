package com.example.smartmirror;

import android.app.ProgressDialog;
import android.content.Context;
import android.content.Intent;
import android.graphics.drawable.Drawable;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.TextView;
import android.widget.Toast;

import androidx.fragment.app.FragmentManager;
import androidx.fragment.app.FragmentTransaction;

import com.bumptech.glide.Glide;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.concurrent.ExecutionException;

// musinsa recommendation

public class ListAdapter extends BaseAdapter {

    private Context context;
    private ArrayList<ListItem> listItems = new ArrayList<ListItem>();

    private String top_url = "null";
    private String outer_url = "null";
    private String bottom_url = "null";
    private String onepiece_url = "null";


    private String top = "null";
    private String outer = "null";
    private String bottom = "null";
    private String onepiece = "null";

    private String ID;
    String styling; // table name

    String rec_case; // musinsa or user's clothes

    FragmentManager fragmentManager;
    private FragmentTransaction transaction;

    public ListAdapter(Context context) {
        this.context = context;
    }

    @Override
    public int getCount() {
        return listItems.size();
    }

    @Override
    public Object getItem(int i) {
        return listItems.get(i);
    }

    @Override
    public long getItemId(int i) {
        return i;
    }


    @Override
    public View getView(int position, View convertView, ViewGroup parent) {

        if (convertView == null) {
            LayoutInflater inflater = (LayoutInflater) context.getSystemService(Context.LAYOUT_INFLATER_SERVICE);
            convertView = inflater.inflate(R.layout.listview_styling, parent, false);
        }

        ListItem listItem = listItems.get(position);
//        final TextView titleTextView = (TextView) convertView.findViewById(R.id.stylingTitle);

        ImageButton doneCheck = (ImageButton) convertView.findViewById(R.id.stylingDetail);
        Glide.with(context).load(listItem.getImage()).override(800, 800).into(doneCheck);
        doneCheck.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                //Intent intent = new Intent(Intent.ACTION_VIEW, Uri.parse(listItem.getLink()));
                Intent intent = new Intent(context, StylingItemActivity.class);
                //intent.putExtra("count",count);
                //Log.e("listadapter-id",listItem.getID());
                intent.putExtra("ID", listItem.getID());
                //Log.e("listadapter-styling", styling);
                intent.putExtra("styling", styling);

                context.startActivity(intent);
            }
        });

        //virtual fitting
        Button fitting = (Button) convertView.findViewById(R.id.virtualFitting);
        fitting.setOnClickListener(new View.OnClickListener() {
            public void onClick(View view) {
                String id = listItem.getID();
                String address="http://52.79.59.24/fitting.php?"; // TODO: 테스트용
                //String address="http://52.79.59.24/virtualFitting.php?STYLING="+styling+"&ID="+id; TODO: 실제 사용해야하는 코드

                virtualfittingThread fitting_thread = new virtualfittingThread();
//
                //virtualfittingThread fitting = ThreadHandler.getThread();
                //fitting.execute("lin"+address);
                try {
                    String result=fitting_thread.execute("lin"+address).get();
                    Log.e("[socket result]",result);
                    if(result.equals("fitting"))
                    {
                        Log.e("[toast]","fitting");
                        Toast.makeText(context.getApplicationContext(), "스마트미러로 스타일링을 확인해보세요!", Toast.LENGTH_LONG).show();
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
//                ThreadHandler.setThread(fitting_thread);
                //ThreadHandler.execute("lin"+address);
                //fitting_thread.execute("lin"+address); //lin을 보내면 라즈베리파이에서 링크 실행하게끔
                // fitting_thread.start();
                //TODO

            }
        });

        return convertView;
    }

    public void addItem(String url, String Link, String ID) {
        ListItem listItem = new ListItem();

        listItem.setImage(url);
        listItem.setLink(Link);
        listItem.setID(ID);
        listItems.add(listItem);
    }

    public void setTopImage(String top_url, String top) {
        this.top_url = top_url;
        this.top = top;
    }

    public void setBottomImage(String bottom_url, String bottom) {
        this.bottom_url = bottom_url;
        this.bottom = bottom;
    }

    public void setOuterImage(String outer_url, String outer) {
        this.outer_url = outer_url;
        this.outer = outer;
    }

    public void setOnepieceImage(String onepiece_url, String onepiece) {
        this.onepiece_url = onepiece_url;
        this.onepiece = onepiece;
    }

    public void setID(String ID) {
        this.ID = ID;
    }

    public void setStyling(String styling) {
        this.styling = styling;
    }

    public void removeItem(int pos) {
        listItems.remove(pos);
        notifyDataSetChanged();
    }
    public void setRec_case(String rec_case)
    {
        this.rec_case=rec_case;
    }
    public String getRec_case()
    {
        return rec_case;
    }


}
