package com.google.firebase.samples.apps.mlkit.java.customobjectdetection;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Rect;
import android.os.SystemClock;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.ml.common.FirebaseMLException;
import com.google.firebase.ml.vision.FirebaseVision;
import com.google.firebase.ml.vision.common.FirebaseVisionImage;
import com.google.firebase.ml.vision.objects.FirebaseVisionObject;
import com.google.firebase.ml.vision.objects.FirebaseVisionObjectDetector;
import com.google.firebase.ml.vision.objects.FirebaseVisionObjectDetectorOptions;
import com.google.firebase.samples.apps.mlkit.common.CameraImageGraphic;
import com.google.firebase.samples.apps.mlkit.common.FrameMetadata;
import com.google.firebase.samples.apps.mlkit.common.GraphicOverlay;
import com.google.firebase.samples.apps.mlkit.java.VisionProcessorBase;
import com.google.firebase.samples.apps.mlkit.java.custommodel.CustomImageClassifier;
import com.google.firebase.samples.apps.mlkit.java.custommodel.CustomImageClassifierProcessor;
import com.google.firebase.samples.apps.mlkit.java.objectdetection.ObjectGraphic;

import java.io.IOException;
import java.lang.ref.Reference;
import java.lang.ref.WeakReference;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

public class CustomObjectDetectorProcessor extends VisionProcessorBase<List<FirebaseVisionObject>> {

    private static final String TAG = "ObjectDetectorProcessor";

    private final FirebaseVisionObjectDetector detector;
    private final CustomImageClassifier classifier;
    private final Reference<Activity> activityRef;
    private Boolean useQuantizedModel;

    public CustomObjectDetectorProcessor (final FirebaseVisionObjectDetectorOptions options, final Activity activity, final boolean useQuantizedModel) throws FirebaseMLException {
        detector = FirebaseVision.getInstance().getOnDeviceObjectDetector(options);
        activityRef = new WeakReference<>(activity);
        classifier = new CustomImageClassifier(activity.getApplicationContext(), useQuantizedModel);
        this.useQuantizedModel=useQuantizedModel;
    }

    @Override
    public void stop() {
        super.stop();
        try {
            detector.close();
        } catch (IOException e) {
            Log.e(TAG, "Exception thrown while trying to close object detector!", e);
        }
    }

    @Override
    protected Task<List<FirebaseVisionObject>> detectInImage(FirebaseVisionImage image) {
        return detector.processImage(image);
    }

    @Override
    protected void onSuccess(
            @Nullable Bitmap originalCameraImage,
            @NonNull List<FirebaseVisionObject> results,
            @NonNull FrameMetadata frameMetadata,
            @NonNull GraphicOverlay graphicOverlay) throws FirebaseMLException {
        for (FirebaseVisionObject object : results) {
            processFirebaseVisionObject(originalCameraImage, object, graphicOverlay);
        }
    }

    @Override
    protected void onFailure(@NonNull Exception e) {
        Log.e(TAG, "Object detection failed!", e);
    }

    private void processFirebaseVisionObject(final Bitmap originalCameraImage, final FirebaseVisionObject object, final GraphicOverlay graphicOverlay) throws FirebaseMLException {

        final Bitmap bitmap = getBitmapFromFVO(originalCameraImage,object);
        final ByteBuffer data = ByteBuffer.allocateDirect(bitmap.getByteCount());
        bitmap.copyPixelsToBuffer(data);
        final int width = bitmap.getWidth();
        final int height = bitmap.getHeight();

        final Activity activity = activityRef.get();
        if (activity == null) {
            return;
        }

        classifier
                .classifyFrame(data, width, height)
                .addOnSuccessListener(
                        activity,
                        new OnSuccessListener<List<String>>() {
                            @Override
                            public void onSuccess(final List<String> result) {

                                CustomObjectGraphic objectGraphic=new CustomObjectGraphic(graphicOverlay,object,result);
                                CameraImageGraphic imageGraphic =
                                        new CameraImageGraphic(graphicOverlay, originalCameraImage);
                                graphicOverlay.clear();
                                graphicOverlay.add(imageGraphic);
                                graphicOverlay.add(objectGraphic);
                                graphicOverlay.postInvalidate();
                            }
                        })
                .addOnFailureListener(
                        new OnFailureListener() {
                            @Override
                            public void onFailure(@NonNull Exception e) {
                                Log.d(TAG, "Custom classifier failed: " + e);
                                e.printStackTrace();
                            }
                        });

    }

    private Bitmap getBitmapFromFVO (Bitmap bitmap, FirebaseVisionObject object){
        Rect rect = object.getBoundingBox();
        int w = rect.right - rect.left;
        int h = rect.bottom - rect.top;
        Bitmap ret = Bitmap.createBitmap(w, h, bitmap.getConfig());
        Canvas canvas = new Canvas(ret);
        canvas.drawBitmap(bitmap, -rect.left, -rect.top, null);
        return ret;
    }
}
