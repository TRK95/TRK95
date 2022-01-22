package com.example.imagepro;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.hardware.Camera;
import android.os.Build;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.Toast;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.text.Text;
import com.google.mlkit.vision.text.TextRecognition;
import com.google.mlkit.vision.text.TextRecognizer;
import com.google.mlkit.vision.text.latin.TextRecognizerOptions;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.time.LocalDateTime;
import java.util.Locale;

public class CameraActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2{
    private static final String TAG="MainActivity";
    private TextToSpeech mTTS;
    private Mat mRgba;
    private Mat mGray;
    private CameraBridgeViewBase mOpenCvCameraView;

    private age_genderDetector ageGenderDetector;
    private objectDetectorClass Objectdetectorclass;
    private expressionDetector expressiondetector;
    private Bitmap bitmap=null;
    private TextRecognizer textRecognizer;
    private Button expression;
    private Button text_;
    private Button object;
    private Button person;
    private Button flip_camera;
    private int second_start;
    private int second_end;

    private double distance = 0;
    private double image_height;
    private double real_height= (double) 0.14;
    private Camera mCamera;
    private double focal_length;
    private double sensor_height;
    private double object_height;

    private int mCameraID=0;

    private String speak="hello";
    private String change="object";

    private BaseLoaderCallback mLoaderCallback =new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status){
                case LoaderCallbackInterface
                        .SUCCESS:{
                    Log.i(TAG,"OpenCv Is loaded");
                    mOpenCvCameraView.enableView();
                }
                default:
                {
                    super.onManagerConnected(status);

                }
                break;
            }
        }
    };

    public CameraActivity(){
        Log.i(TAG,"Instantiated new "+this.getClass());
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);


        int MY_PERMISSIONS_REQUEST_CAMERA=0;
        // if camera permission is not given it will ask for it on device
        if (ContextCompat.checkSelfPermission(CameraActivity.this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_DENIED){
            ActivityCompat.requestPermissions(CameraActivity.this, new String[] {Manifest.permission.CAMERA}, MY_PERMISSIONS_REQUEST_CAMERA);
        }




        setContentView(R.layout.activity_camera);
        //imageButton=(ImageButton) findViewById(R.id.btnspeak);
        text_=(Button) findViewById(R.id.text);
        person=(Button) findViewById(R.id.person);
        object=(Button) findViewById(R.id.object);
        expression=(Button) findViewById(R.id.expression);
        mOpenCvCameraView=(CameraBridgeViewBase) findViewById(R.id.frame_Surface);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        flip_camera=(Button) findViewById(R.id.flip);
        // get image_height in pixel






        try{
            // input size is 300 for this model

            mCamera=Camera.open();
            DisplayMetrics metrics = new DisplayMetrics();
            getWindowManager().getDefaultDisplay().getMetrics(metrics);

            //focal_length metres
            Camera.Parameters params = mCamera.getParameters();
            focal_length = params.getFocalLength()/1000;//0.045m
            //sensor_height
            double a=params.getVerticalViewAngle();
            a = Math.toRadians(a);
            sensor_height=Math.tan(a/2)*2*focal_length;
            expressiondetector = new expressionDetector(getAssets(), CameraActivity.this,"expression_model.tflite",48);
            textRecognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS);
            ageGenderDetector = new age_genderDetector(getAssets(), CameraActivity.this, "model.tflite",96);
            Objectdetectorclass=new objectDetectorClass(getAssets(),"ssd_mobilenet.tflite","labelmap.txt",300);
            Log.d("MainActivity","Model is successfully loaded");
        }
        catch (IOException e){
            Log.d("MainActivity","Getting some error");
            e.printStackTrace();
        }
        flip_camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                swapCamera();
            }
        });



        text_.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                change = "text";
                expressiondetector.val="click button again";
                Mat a=mRgba.t();
                Core.flip(a,mRgba,1);
                bitmap=Bitmap.createBitmap(mRgba.cols(),mRgba.rows(),Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(mRgba,bitmap);
                InputImage image=InputImage.fromBitmap(bitmap,0);
                Task<Text> result = textRecognizer.process(image)
                        .addOnSuccessListener(new OnSuccessListener<Text>() {
                            @Override
                            public void onSuccess(Text text) {
                                speak = text.getText();
                                mTTS.speak(speak, TextToSpeech.QUEUE_FLUSH, null);
                                Log.d("textDetectorClass",  text.getText());

                            }
                        })
                        .addOnFailureListener(new OnFailureListener() {
                            @Override
                            public void onFailure(@NonNull Exception e) {
                                //     speak="can't detect any text";
                            }
                        });

            }
        });
        object.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                change = "object";
                expressiondetector.val="click button again";
            }
        });
        person.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                change = "person";
                expressiondetector.val="click button again";

                object_height=ageGenderDetector.object_height;
                Log.d("object height", "objectheight: " +sensor_height);
                if(object_height!=0 && sensor_height!= 0){
                    distance=(focal_length*real_height*image_height)/(object_height*sensor_height);
                    distance=(double) Math.round(distance * 100) / 100;
                }
                if(distance != 0.0 ){
                    mTTS.speak(ageGenderDetector.speak + " is " + ageGenderDetector.agespk +", distance is" + distance + "metres" , TextToSpeech.QUEUE_FLUSH, null);
                } else {
                    mTTS.speak(ageGenderDetector.speak + " is " + ageGenderDetector.agespk +", can not measure distance", TextToSpeech.QUEUE_FLUSH, null);

                }
            }
        });
        expression.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                change = "expression";
                mTTS.speak(expressiondetector.val, TextToSpeech.QUEUE_FLUSH, null);
            }
        });

        mTTS = new TextToSpeech(getApplicationContext(), new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if (status != TextToSpeech.ERROR){
                    mTTS.setLanguage(Locale.ENGLISH);
                } else {
                    Toast.makeText(CameraActivity.this, "Error", Toast.LENGTH_SHORT).show();
                }
            }
        });

    }

    private void swapCamera() {
        mCameraID=mCameraID^1;
        mOpenCvCameraView.disableView();
        mOpenCvCameraView.setCameraIndex(mCameraID);
        mOpenCvCameraView.enableView();
    }

    //@Override
    //protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
    //  super.onActivityResult(requestCode, resultCode, data);

    //if(requestCode == 100 && resultCode == RESULT_OK){
    //  change = data.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS).get(0).toString();
    //}
    //}

    @Override
    protected void onResume() {
        super.onResume();
        if (OpenCVLoader.initDebug()){
            //if load success
            Log.d(TAG,"Opencv initialization is done");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
        else{
            //if not loaded
            Log.d(TAG,"Opencv is not loaded. try again");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0,this,mLoaderCallback);
        }
    }


    @Override
    protected void onPause() {
        super.onPause();
        if (mOpenCvCameraView !=null){
            mOpenCvCameraView.disableView();
        }
    }

    public void onDestroy(){
        super.onDestroy();
        if(mOpenCvCameraView !=null){
            mOpenCvCameraView.disableView();
            mTTS.stop();

            //result.stop();
        }

    }

    public void onCameraViewStarted(int width ,int height){
        mRgba=new Mat(height,width, CvType.CV_8UC4);
        mGray =new Mat(height,width,CvType.CV_8UC1);
    }
    public void onCameraViewStopped(){
        mRgba.release();
    }
    @RequiresApi(api = Build.VERSION_CODES.O)
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame){
        mRgba=inputFrame.rgba();
        mGray=inputFrame.gray();

        if(mCameraID == 1){
            Core.flip(mRgba,mRgba,-1);
            Core.flip(mGray,mGray,-1);
        }

        if(change == "person"){
            mRgba=ageGenderDetector.recognizeImage(mRgba);
            Log.d("check camera slip","successful");
        }

        // Before watching this video please watch previous video of loading tensorflow lite model
        if (change == "expression") {
            mRgba=expressiondetector.recognizeImage(mRgba);
        }

        if (change == "object") {
            image_height=Objectdetectorclass.height;
            mRgba = Objectdetectorclass.recognizeImage(mRgba);

            if(Objectdetectorclass.speak != Objectdetectorclass.speak1){
                mTTS.speak(Objectdetectorclass.speak, TextToSpeech.QUEUE_FLUSH, null);
                Objectdetectorclass.speak1=Objectdetectorclass.speak;
                LocalDateTime start = LocalDateTime.now();
                second_start= start.toLocalTime().toSecondOfDay();
            }
            else {
                LocalDateTime end = LocalDateTime.now();
                second_end= end.toLocalTime().toSecondOfDay();
                if((second_end-second_start)==5){
                    mTTS.speak(Objectdetectorclass.speak, TextToSpeech.QUEUE_FLUSH, null);
                }
            }



        }

        //Mat out=new Mat();
        //out=objectDetectorClass.recognizeImage(mRgba);
        //out=ageGenderDetector.recognizeImage(mRgba);
        return mRgba;
    }

}