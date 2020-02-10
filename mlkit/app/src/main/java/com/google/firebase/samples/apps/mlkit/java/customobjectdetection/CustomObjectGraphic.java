package com.google.firebase.samples.apps.mlkit.java.customobjectdetection;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.Log;

import com.google.firebase.ml.vision.objects.FirebaseVisionObject;
import com.google.firebase.samples.apps.mlkit.common.GraphicOverlay;

import java.util.List;

public class CustomObjectGraphic extends GraphicOverlay.Graphic {

    private static final String TAG = "CustObjGraphic";
    private static final float TEXT_SIZE = 54.0f;
    private static final float STROKE_WIDTH = 4.0f;

    private final FirebaseVisionObject object;
    private final List<String> results;
    private final Paint boxPaint;
    private final Paint textPaint;

    public CustomObjectGraphic (final GraphicOverlay overlay, final FirebaseVisionObject object, final List<String> results) {
        super(overlay);

        this.object = object;
        this.results = results;

        boxPaint = new Paint();
        boxPaint.setColor(Color.WHITE);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(STROKE_WIDTH);

        textPaint = new Paint();
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(TEXT_SIZE);
    }

    @Override
    public void draw(Canvas canvas) {
        // Draws the bounding box.
        RectF rect = new RectF(object.getBoundingBox());
        rect.left = translateX(rect.left);
        rect.top = translateY(rect.top);
        rect.right = translateX(rect.right);
        rect.bottom = translateY(rect.bottom);
        canvas.drawRect(rect, boxPaint);

        // Draws other object info.
        canvas.drawText(results.get(0), rect.left, rect.bottom, textPaint);
        /*
        canvas.drawText("trackingId: " + object.getTrackingId(), rect.left, rect.top, textPaint);
        canvas.drawText(
                "confidence: " + object.getClassificationConfidence(), rect.right, rect.bottom, textPaint);*/
    }

}
