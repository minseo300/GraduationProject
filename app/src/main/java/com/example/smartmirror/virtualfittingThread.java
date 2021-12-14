package com.example.smartmirror;

import android.os.AsyncTask;
import android.util.Log;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.net.InetAddress;
import java.net.Socket;
import java.net.UnknownHostException;
import java.nio.charset.StandardCharsets;
import java.util.logging.Handler;

//////////////////////////////////////////////////////////////////////////////

public class virtualfittingThread extends AsyncTask<String,String,String>{
    private String output_message;
    private String input_message;
    Socket socket;
    private static String ip="192.9.116.211";
//    private static String ip="192.9.116.162";
    //private static String ip = "192.9.116.43";
    private int port;
    private DataOutputStream dataOutput;
    private DataInputStream dataInput;
    String msg;
    String result;
//    public virtualfittingThread()
//    {
//        try {
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//    }
    protected void onPreExecute()
    {
        super.onPreExecute();
    }
    protected String doInBackground(String... strings){
        try{
            Log.e("conneting...", "dfd");
           if (SocketHandler.getSocket()!=null)
           {
               socket=SocketHandler.getSocket();

               Log.e("tt", "ttt");
           }
           else{
               Log.e("else", "~~~");
               socket = new Socket(ip,9990);
               Log.e("make", "~~~");
               SocketHandler.setSocket(socket);
               Log.e("~~~", "~~~");
           }
            Log.e("[socket]","before");
            //InetAddress serverAddress=InetAddress.getByName(ip);
            //Log.e("serverAddress",String.valueOf(ip));
            //socket=new Socket(serverAddress,9999);

            Boolean result=socket.isConnected();
            Log.e("[socket]",String.valueOf(result));

            Log.e("[socket]","after");
            dataOutput=new DataOutputStream(socket.getOutputStream());
            dataInput=new DataInputStream(socket.getInputStream());
            output_message=strings[0];
            dataOutput.writeUTF(output_message);
            Log.e("send", output_message);
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }


        try
        {
            Log.e("ouput",output_message);
            byte[] buffer=new byte[10000000];
            int read_Byte=dataInput.read(buffer);
            input_message=new String(buffer,0,read_Byte);
//            if(!input_message.equals("stop"))
//            {
//                publishProgress(input_message);
//            }
//            else //stop message
//            {
//
//            }
            if(input_message.equals("fitting"))
            {
                result="fitting";
            }
            else if(input_message.equals("camera"))
            {
                result="camera";
            }
            else if(input_message.startsWith("recommended"))
            {
                Log.e("rec input message",input_message);
                String[] recommendation=input_message.split(":");
                result=recommendation[1];
                Log.e("rec result",result);
                //result="recommended";
            }
            // Thread.sleep(2);
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }

        return result;
    }
    protected void onProgressUpdate(String... params) // 스레드가 수행되는 사이에 수행할 중간작업(메인 스레드) -doInBackground()에서 publishProgress()메소드를 호출하여 중간 작업 수행 가능
    {
        Log.e("[PROGRESS]","소켓 통신 중 ,,,");
    }
    protected void onPostExecute(String result) // 스레드 작업이 모두 끝난 후에 수행할 작업(메인스레드)
    {
        Log.e("[SUCCESS]","소켓 작업 완료");
//        if(socket!=null) {
//            try {
//                socket.close();
//            } catch (Exception e) {
//                e.printStackTrace();
//            }
//        }

        //TODO
    }
}


