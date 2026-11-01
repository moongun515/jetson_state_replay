package com.moongun.fallguard;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.media.MediaMetadataRetriever;
import android.net.Uri;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.Collections;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

/** Runs the YOLO26n-pose checkpoint and temporal classifier fully on device. */
final class PoseAnalyzer {
    private static final int SIZE = 640;
    private static final int FRAMES = 12;
    private static final int JOINTS = 17;
    private static final float MIN_PERSON_SCORE = 0.15f;

    static final class Outcome {
        final double probability;
        final double threshold;
        final int foundFrames;
        final long durationMs;

        Outcome(double probability, double threshold, int foundFrames, long durationMs) {
            this.probability = probability;
            this.threshold = threshold;
            this.foundFrames = foundFrames;
            this.durationMs = durationMs;
        }
    }

    static Outcome analyze(Context context, Uri video) throws Exception {
        PoseClassifier classifier;
        try (InputStream stream = context.getAssets().open("classifier.json")) {
            classifier = new PoseClassifier(stream);
        }
        byte[] model;
        try (InputStream stream = context.getAssets().open("yolo26n-pose.onnx")) {
            ByteArrayOutputStream bytes = new ByteArrayOutputStream();
            byte[] chunk = new byte[65536];
            int n;
            while ((n = stream.read(chunk)) != -1) bytes.write(chunk, 0, n);
            model = bytes.toByteArray();
        }

        OrtEnvironment environment = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setIntraOpNumThreads(2);
        MediaMetadataRetriever retriever = new MediaMetadataRetriever();
        try (OrtSession session = environment.createSession(model, options)) {
            retriever.setDataSource(context, video);
            String durationText = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION);
            if (durationText == null) throw new IllegalArgumentException("영상 길이를 읽을 수 없습니다.");
            long durationMs = Long.parseLong(durationText);
            if (durationMs < 1000 || durationMs > 30000) {
                throw new IllegalArgumentException("1~30초로 잘라 둔 영상을 선택하세요.");
            }
            String inputName = session.getInputNames().iterator().next();
            float[][] sequence = new float[FRAMES][56];
            int found = 0;
            for (int index = 0; index < FRAMES; index++) {
                long timeUs = Math.round((double) index * Math.max(0, durationMs - 50) * 1000 / (FRAMES - 1));
                Bitmap frame = retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST);
                if (frame == null) throw new IllegalArgumentException("영상 프레임을 읽을 수 없습니다.");
                try {
                    int width = frame.getWidth(), height = frame.getHeight();
                    float ratio = Math.min(SIZE / (float) width, SIZE / (float) height);
                    int resizedWidth = Math.round(width * ratio), resizedHeight = Math.round(height * ratio);
                    int left = Math.round((SIZE - resizedWidth) / 2f - 0.1f);
                    int top = Math.round((SIZE - resizedHeight) / 2f - 0.1f);
                    float[] input = letterboxInput(frame, left, top, resizedWidth, resizedHeight);
                    try (OnnxTensor tensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(input),
                            new long[]{1, 3, SIZE, SIZE});
                         OrtSession.Result result = session.run(Collections.singletonMap(inputName, tensor))) {
                        float[][][] output = (float[][][]) result.get(0).getValue();
                        if (output.length != 1 || output[0].length != 300 || output[0][0].length != 57) {
                            throw new IllegalStateException("Unexpected YOLO26 pose output shape");
                        }
                        if (extractFrameFeatures(output[0], width, height, ratio, left, top, sequence[index])) found++;
                    }
                } finally {
                    frame.recycle();
                }
            }
            return new Outcome(classifier.probability(sequence), classifier.threshold, found, durationMs);
        } finally {
            retriever.release();
            options.close();
        }
    }

    private static float[] letterboxInput(Bitmap frame, int left, int top, int width, int height) {
        Bitmap canvasBitmap = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888);
        try {
            Canvas canvas = new Canvas(canvasBitmap);
            canvas.drawColor(Color.rgb(114, 114, 114));
            Paint paint = new Paint(Paint.FILTER_BITMAP_FLAG);
            canvas.drawBitmap(frame, new Rect(0, 0, frame.getWidth(), frame.getHeight()),
                    new RectF(left, top, left + width, top + height), paint);
            int[] pixels = new int[SIZE * SIZE];
            canvasBitmap.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE);
            float[] input = new float[3 * SIZE * SIZE];
            for (int i = 0; i < pixels.length; i++) {
                input[i] = Color.red(pixels[i]) / 255f;
                input[SIZE * SIZE + i] = Color.green(pixels[i]) / 255f;
                input[2 * SIZE * SIZE + i] = Color.blue(pixels[i]) / 255f;
            }
            return input;
        } finally {
            canvasBitmap.recycle();
        }
    }

    private static float clamp(float value, float low, float high) {
        return Math.max(low, Math.min(high, value));
    }

    private static boolean extractFrameFeatures(float[][] rows, int imageWidth, int imageHeight,
                                                 float ratio, int left, int top, float[] features) {
        int chosen = -1;
        float best = -1;
        float[] chosenBox = new float[4];
        for (int i = 0; i < rows.length; i++) {
            float[] row = rows[i];
            if (row[4] < MIN_PERSON_SCORE || Math.round(row[5]) != 0) continue;
            float x1 = clamp((row[0] - left) / ratio, 0, imageWidth);
            float y1 = clamp((row[1] - top) / ratio, 0, imageHeight);
            float x2 = clamp((row[2] - left) / ratio, 0, imageWidth);
            float y2 = clamp((row[3] - top) / ratio, 0, imageHeight);
            float area = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
            float rank = row[4] * (float) Math.sqrt(Math.max(area, 1));
            if (rank > best) {
                chosen = i;
                best = rank;
                chosenBox[0] = x1;
                chosenBox[1] = y1;
                chosenBox[2] = x2;
                chosenBox[3] = y2;
            }
        }
        if (chosen < 0) return false;
        float[] row = rows[chosen];
        float width = Math.max(chosenBox[2] - chosenBox[0], 1);
        float height = Math.max(chosenBox[3] - chosenBox[1], 1);
        float centerX = (chosenBox[0] + chosenBox[2]) / 2;
        float centerY = (chosenBox[1] + chosenBox[3]) / 2;
        float size = Math.max(width, height);
        for (int joint = 0; joint < JOINTS; joint++) {
            int outputIndex = 6 + 3 * joint;
            float confidence = clamp(row[outputIndex + 2], 0, 1);
            if (confidence >= 0.1f) {
                float x = clamp((row[outputIndex] - left) / ratio, 0, imageWidth);
                float y = clamp((row[outputIndex + 1] - top) / ratio, 0, imageHeight);
                features[3 * joint] = (x - centerX) / size;
                features[3 * joint + 1] = (y - centerY) / size;
            }
            features[3 * joint + 2] = confidence;
        }
        features[51] = (float) Math.log(width / height);
        features[52] = centerY / imageHeight;
        features[53] = chosenBox[3] / imageHeight;
        features[54] = width * height / (imageWidth * imageHeight);
        features[55] = row[4];
        return true;
    }
}
