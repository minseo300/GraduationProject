package com.example.smartmirror;

import android.annotation.SuppressLint;
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


public class StylingitemFragment extends Fragment {

    ListView itemList;
    ListItem items;
    StylingItemAdapter adapter;


    private String top_url; //이미지 주소
    private String outer_url;
    private String bottom_url;
    private String onepiece_url;


    private String top; // 판매링크
    private String outer;
    private String bottom;
    private String onepiece;

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        Bundle bundle=this.getArguments();
        top_url=bundle.getString("top_url");
        outer_url=bundle.getString("outer_url");
        bottom_url=bundle.getString("bottom_url");
        onepiece_url=bundle.getString("onepiece_url");

        top=bundle.getString("top");
        outer=bundle.getString("outer");
        bottom=bundle.getString("bottom");
        onepiece=bundle.getString("onepiece");
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_stylingitem, container, false);
    }
    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        itemList = view.findViewById(R.id.ItemList);
        items = new ListItem();
        adapter = new StylingItemAdapter(getContext());

        if(top_url!=null)
        {
            adapter.addItem(top_url,top);
        }
        if(outer_url!=null)
        {
            adapter.addItem(outer_url,outer);
        }
        if(bottom_url!=null)
        {
            adapter.addItem(bottom_url,bottom);
        }
        if(onepiece_url!=null)
        {
            adapter.addItem(onepiece_url,onepiece);
        }

        itemList.setAdapter(adapter);


    }
}