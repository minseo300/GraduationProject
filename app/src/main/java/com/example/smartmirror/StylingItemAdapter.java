package com.example.smartmirror;

import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.Button;
import android.widget.ImageButton;

import com.bumptech.glide.Glide;

import java.util.ArrayList;

public class StylingItemAdapter extends BaseAdapter {

    private Context context;
    private ArrayList<ListItem> listItems = new ArrayList<ListItem>();

    String styling;
    String ID;
    public StylingItemAdapter(Context context){
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

        if(convertView == null){
            LayoutInflater inflater = (LayoutInflater) context.getSystemService(Context.LAYOUT_INFLATER_SERVICE);
            convertView = inflater.inflate(R.layout.listview_stylingitem, parent, false);
        }

        ListItem listItem = listItems.get(position);

        ImageButton doneCheck = (ImageButton) convertView.findViewById(R.id.stylingLink);
        Glide.with(context).load(listItem.getImage()).override(800,800).into(doneCheck);
        doneCheck.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                //Log.e("link: ",listItem.getLink());
                Intent intent = new Intent(Intent.ACTION_VIEW, Uri.parse(listItem.getLink()));
                intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
                context.startActivity(intent);
            }
        });
        return convertView;
    }

    public void addItem(String url, String Link){
        ListItem listItem = new ListItem();

        listItem.setImage(url);
        listItem.setLink(Link);
        listItems.add(listItem);
    }

    public void removeItem(int pos){
        listItems.remove(pos);
        notifyDataSetChanged();
    }

    public void setStyling(String styling)
    {
        this.styling=styling;
    }
    public void setID(String ID)
    {
        this.ID=ID;
    }
}
