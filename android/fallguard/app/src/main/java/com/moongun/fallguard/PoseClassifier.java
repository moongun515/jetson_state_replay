package com.moongun.fallguard;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

/** Portable copy of the Python StandardScaler + logistic regression pipeline. */
final class PoseClassifier {
    private static final int FRAMES = 12;
    private static final int FRAME_FEATURES = 56;
    private static final int SUMMARY_FEATURES = 392;

    private final double[] mean;
    private final double[] scale;
    private final double[] coefficients;
    private final double intercept;
    final double threshold;

    PoseClassifier(InputStream stream) throws Exception {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        byte[] buffer = new byte[8192];
        int n;
        while ((n = stream.read(buffer)) != -1) bytes.write(buffer, 0, n);
        JSONObject json = new JSONObject(bytes.toString(StandardCharsets.UTF_8.name()));
        if (!"fall-pose-logistic-v1".equals(json.getString("schema"))) {
            throw new IllegalArgumentException("Unknown classifier schema");
        }
        mean = readArray(json.getJSONArray("mean"));
        scale = readArray(json.getJSONArray("scale"));
        coefficients = readArray(json.getJSONArray("coefficients"));
        intercept = json.getDouble("intercept");
        threshold = json.getDouble("threshold");
    }

    private static double[] readArray(JSONArray input) throws Exception {
        if (input.length() != SUMMARY_FEATURES) throw new IllegalArgumentException("Wrong feature count");
        double[] result = new double[SUMMARY_FEATURES];
        for (int i = 0; i < result.length; i++) result[i] = input.getDouble(i);
        return result;
    }

    double probability(float[][] sequence) {
        if (sequence.length != FRAMES) throw new IllegalArgumentException("Expected 12 frames");
        double[] summary = new double[SUMMARY_FEATURES];
        for (int feature = 0; feature < FRAME_FEATURES; feature++) {
            double total = 0;
            double minimum = Double.POSITIVE_INFINITY;
            double maximum = Double.NEGATIVE_INFINITY;
            for (float[] frame : sequence) {
                if (frame.length != FRAME_FEATURES) throw new IllegalArgumentException("Expected 56 features");
                double value = frame[feature];
                total += value;
                minimum = Math.min(minimum, value);
                maximum = Math.max(maximum, value);
            }
            double average = total / FRAMES;
            double variance = 0;
            for (float[] frame : sequence) {
                double delta = frame[feature] - average;
                variance += delta * delta;
            }
            summary[feature] = average;
            summary[FRAME_FEATURES + feature] = Math.sqrt(variance / FRAMES);
            summary[2 * FRAME_FEATURES + feature] = minimum;
            summary[3 * FRAME_FEATURES + feature] = maximum;
            summary[4 * FRAME_FEATURES + feature] = sequence[0][feature];
            summary[5 * FRAME_FEATURES + feature] = sequence[FRAMES - 1][feature];
            summary[6 * FRAME_FEATURES + feature] = sequence[FRAMES - 1][feature] - sequence[0][feature];
        }
        double logit = intercept;
        for (int i = 0; i < SUMMARY_FEATURES; i++) {
            logit += coefficients[i] * ((summary[i] - mean[i]) / scale[i]);
        }
        return 1.0 / (1.0 + Math.exp(-logit));
    }
}
