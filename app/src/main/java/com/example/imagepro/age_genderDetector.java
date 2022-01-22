package com.example.imagepro;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class age_genderDetector {
    private Interpreter interpreter;
    // store all label in array
    private List<String> labelList;
    private int INPUT_SIZE;
    private int IMAGE_MEAN=0;
    private  float IMAGE_STD=255.0f;
    // use to initialize gpu in app
    private GpuDelegate gpuDelegate=null;
    private int height=0;
    private  int width=0;

    float image_height;
    int object_height;
    private CameraActivity cameraActivity;

    private CascadeClassifier cascadeClassifier;
    String agespk="adult";
    String speak="Click button again";



    age_genderDetector(AssetManager assetManager, Context context, String modelPath, int inputSize) throws IOException{
        INPUT_SIZE=inputSize;
        // use to define gpu or cpu // no. of threads
        Interpreter.Options options=new Interpreter.Options();
        gpuDelegate=new GpuDelegate();
        options.addDelegate(gpuDelegate);
        options.setNumThreads(4); // set it according to your phone
        // loading model
        interpreter=new Interpreter(loadModelFile(assetManager,modelPath),options);
        Log.d("age_gender success", "CNN model is loaded");
        // load carras
        try {
            InputStream is=context.getResources().openRawResource(R.raw.haarcascade_frontalface_alt);

            File cascadedir=context.getDir("cascade", Context.MODE_PRIVATE);
            File mCascadeFile=new File(cascadedir,"haarcascade_frontalface_alt");
            FileOutputStream os=new FileOutputStream(mCascadeFile);

            byte[] buffer=new byte[4096];
            int byteRead;
            while ((byteRead=is.read(buffer)) != -1){
                os.write(buffer,0,byteRead);
            }
            is.close();
            os.close();
            cascadeClassifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
            Log.d("age_gender","haar cascade Classifier is Loaded");
        } catch (IOException e){
            e.printStackTrace();
        }
    }

    public Mat recognizeImage(Mat mat_image){


        Mat a=mat_image.t();
        Core.flip(a,mat_image,1);
        a.release();
        // if you do not do this process you will get improper prediction, less no. of object
        Mat grayscaleImage=new Mat();
        Imgproc.cvtColor(mat_image,grayscaleImage,Imgproc.COLOR_RGB2GRAY);
        height=grayscaleImage.height();
        width=grayscaleImage.width();
        int absoluteFaceSize=(int) (height*0.1);
        MatOfRect faces=new MatOfRect();

        if(cascadeClassifier != null){
            cascadeClassifier.detectMultiScale(grayscaleImage,faces,1.1,2,2,new Size(absoluteFaceSize,absoluteFaceSize), new Size());
        }
        Rect[] faceArray=faces.toArray();
        for (int i=0; i<faceArray.length; i++ ){
            Imgproc.rectangle(mat_image,faceArray[i].tl(),faceArray[i].br(),new Scalar(0,225,0,225),2);
            // crop the face frome frame
            // starting x coordinates                   starting y coordinate
            Rect roi=new Rect((int)faceArray[i].tl().x,(int)faceArray[i].tl().y,
                    (int)(faceArray[i].br().x)-(int)(faceArray[i].tl().x),
                    (int)faceArray[i].br().y-(int)(faceArray[i].tl().y)
            );




            Mat cropped = new Mat(grayscaleImage,roi);
            Mat cropped_rgba= new Mat(mat_image,roi);
            Bitmap bitmap=null;
            bitmap=Bitmap.createBitmap(cropped_rgba.cols(),cropped_rgba.rows(),Bitmap.Config.ARGB_8888);

            object_height=bitmap.getHeight();







            Utils.matToBitmap(cropped_rgba,bitmap);
            Bitmap scaledBitmap=Bitmap.createScaledBitmap(bitmap,96,96, false);
            ByteBuffer byteBuffer=convertBitmapToByteBuffer(scaledBitmap);

            Object[] input=new Object[1];
            input[0]=byteBuffer;

            Map<Integer,Object> output_map=new TreeMap<>();
            float[][] age=new float[1][1];
            float[][] gender=new float[1][1];
            output_map.put(0,age);
            output_map.put(1,gender);

            interpreter.runForMultipleInputsOutputs(input,output_map);

            Object age_o=output_map.get(0);
            Object gender_o=output_map.get(1);

            int age_value=(int)(float) Array.get(Array.get(age_o,0),0);
            if(age_value<=18){
                agespk="child";
            }
            if(age_value>18&& age_value<=60){
                agespk="adult";
            }
            if(age_value>60){
                agespk="older";
            }
            float gender_value=(float) Array.get(Array.get(gender_o,0),0);
            if(gender_value>0.50){
                Log.d("gender", "My Idol we love you female idol");
                    speak="Female";
                    Imgproc.putText(cropped_rgba,"Female," + agespk ,new Point(10,20) ,1,1.5,new Scalar(255,0,0,255),2);

            } else {
                Imgproc.putText(cropped_rgba,"Male," + agespk,new Point(10,20) ,1,1.5,new Scalar(0,0,255,255),2);
                speak="Male";
            }
            cropped_rgba.copyTo(new Mat(mat_image,roi));
        }




        // before returning rotate back by -90 degree
        Mat b=mat_image.t();
        Core.flip(b,mat_image,0);
        b.release();

        return mat_image;
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer;

        int size_images=96;
        byteBuffer=ByteBuffer.allocateDirect(4*1*size_images*size_images*3);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues=new int[size_images*size_images];
        bitmap.getPixels(intValues,0,bitmap.getWidth(),0,0,bitmap.getWidth(),bitmap.getHeight());
        int pixel=0;

        // some error
        //now run
        for (int i=0;i<size_images;++i){
            for (int j=0;j<size_images;++j){
                final  int val=intValues[pixel++];
                byteBuffer.putFloat((((val>>16)& 0xFF))/255.0f);
                byteBuffer.putFloat((((val>>8)& 0xFF))/255.0f);
                byteBuffer.putFloat(((val& 0xFF))/255.0f);
            }
        }

        return byteBuffer;
    }




    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor=assetManager.openFd(modelPath);
        FileInputStream inputStream=new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel=inputStream.getChannel();
        long startOffset =fileDescriptor.getStartOffset();
        long declaredLength=fileDescriptor.getDeclaredLength();

        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declaredLength);

    }

}
