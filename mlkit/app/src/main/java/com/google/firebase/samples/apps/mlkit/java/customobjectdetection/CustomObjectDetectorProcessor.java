package com.google.firebase.samples.apps.mlkit.java.customobjectdetection;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.Rect;
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

import java.io.IOException;
import java.lang.ref.Reference;
import java.lang.ref.WeakReference;
import java.nio.ByteBuffer;
import java.util.List;

public class CustomObjectDetectorProcessor extends VisionProcessorBase<List<FirebaseVisionObject>> {

    private static final String TAG = "ObjectDetectorProcessor";

    private final FirebaseVisionObjectDetector detector;
    private final CustomImageClassifier classifier;
    private final Reference<Activity> activityRef;

    public CustomObjectDetectorProcessor(final FirebaseVisionObjectDetectorOptions options, final Activity activity, final boolean useQuantizedModel) throws FirebaseMLException {
        detector = FirebaseVision.getInstance().getOnDeviceObjectDetector(options);
        activityRef = new WeakReference<>(activity);
        classifier = new CustomImageClassifier(activity.getApplicationContext(), useQuantizedModel);
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

        final Bitmap bitmap = getBitmapFromFVO(originalCameraImage, object);
        final ByteBuffer data = ByteBuffer.wrap(getNV21(bitmap.getWidth(), bitmap.getHeight(), bitmap));
        final int width = bitmap.getWidth();
        final int height = bitmap.getHeight();

        data.rewind();
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

                                CustomObjectGraphic objectGraphic = new CustomObjectGraphic(graphicOverlay, object, result);
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

    private Bitmap getBitmapFromFVO(Bitmap bitmap, FirebaseVisionObject object) {
        Rect rect = object.getBoundingBox();
        return Bitmap.createBitmap(bitmap, rect.left, rect.top, rect.width(), rect.height());
    }

    private byte[] getNV21(int inputWidth, int inputHeight, Bitmap scaled) {

        int[] argb = new int[inputWidth * inputHeight];

        scaled.getPixels(argb, 0, inputWidth, 0, 0, inputWidth, inputHeight);

        byte[] yuv = new byte[inputHeight * inputWidth + 2 * (int) Math.ceil(inputHeight / 2.0) * (int) Math.ceil(inputWidth / 2.0)];
        encodeYUV420SP(yuv, argb, inputWidth, inputHeight);

        scaled.recycle();

        return yuv;
    }

    private void encodeYUV420SP(byte[] yuv420sp, int[] argb, int width, int height) {
        final int frameSize = width * height;

        int yIndex = 0;
        int uvIndex = frameSize;

        int a, R, G, B, Y, U, V;
        int index = 0;
        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {

                a = (argb[index] & 0xff000000) >> 24; // a is not used obviously
                R = (argb[index] & 0xff0000) >> 16;
                G = (argb[index] & 0xff00) >> 8;
                B = (argb[index] & 0xff) >> 0;

                // well known RGB to YUV algorithm
                Y = ((66 * R + 129 * G + 25 * B + 128) >> 8) + 16;
                U = ((-38 * R - 74 * G + 112 * B + 128) >> 8) + 128;
                V = ((112 * R - 94 * G - 18 * B + 128) >> 8) + 128;

                // NV21 has a plane of Y and interleaved planes of VU each sampled by a factor of 2
                //    meaning for every 4 Y pixels there are 1 V and 1 U.  Note the sampling is every other
                //    pixel AND every other scanline.
                yuv420sp[yIndex++] = (byte) ((Y < 0) ? 0 : ((Y > 255) ? 255 : Y));
                if (j % 2 == 0 && index % 2 == 0) {
                    yuv420sp[uvIndex++] = (byte) ((V < 0) ? 0 : ((V > 255) ? 255 : V));
                    yuv420sp[uvIndex++] = (byte) ((U < 0) ? 0 : ((U > 255) ? 255 : U));
                }

                index++;
            }
        }
    }
}
