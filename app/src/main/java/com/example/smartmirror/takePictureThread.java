package com.example.smartmirror;

import android.os.AsyncTask;
import android.util.Log;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.net.Socket;

public class takePictureThread extends AsyncTask<String,String,Void> {

    private String output_message;
    private String input_message;
    Socket socket;
    private static String ip="192.9.116.221";
    private int port;
    private DataOutputStream dataOutput;
    private DataInputStream dataInput;
    String msg;

    protected Void doInBackground(String... strings){
        try{
            Log.e("[picture socket]","before");
            //InetAddress serverAddress=InetAddress.getByName(ip);
            //Log.e("serverAddress",String.valueOf(ip));
            socket=new Socket(ip,9990);
            //socket=new Socket(serverAddress,9999);

            Boolean result=socket.isConnected();
            Log.e("[picture socket result]",String.valueOf(result));

            Log.e("[picture socket]","after");
            dataOutput=new DataOutputStream(socket.getOutputStream());
            dataInput=new DataInputStream(socket.getInputStream());
            output_message=strings[0];
            dataOutput.writeUTF(output_message);
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }

        while(true)
        {
            try
            {
                Log.e("ouput",output_message);
                byte[] buffer=new byte[2048];
                int read_Byte=dataInput.read(buffer);
                input_message=new String(buffer,0,read_Byte);
                if(!input_message.equals("stop"))
                {
                    publishProgress(input_message);
                }
                else //stop message
                {
                    Log.e("[picture socket]","STOP");
                    break;
                }
                //Thread.sleep(2);
            }
            catch(Exception e)
            {
                e.printStackTrace();
            }
        }
        return null;
    }

    protected void onProgressUpdate(String... params) // 스레드가 수행되는 사이에 수행할 중간작업(메인 스레드) -doInBackground()에서 publishProgress()메소드를 호출하여 중간 작업 수행 가능
    {
        Log.e("[SUCCESS1]","소켓통신중,,");
    }
    protected void onPostExecute(Void result) // 스레드 작업이 모두 끝난 후에 수행할 작업(메인스레드)
    {
        Log.e("[SUCCESS2]","촬영 완료");
        if(socket!=null)
        {
            try{
                socket.close();
            }catch(Exception e)
            {
                e.printStackTrace();
            }
        }
    }
}
