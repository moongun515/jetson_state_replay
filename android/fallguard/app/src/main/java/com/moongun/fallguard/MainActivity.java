package com.moongun.fallguard;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class MainActivity extends Activity {
    private static final int PICK_VIDEO = 1;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private TextView status;
    private Button selectVideo;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        setContentView(R.layout.activity_main);
        status = findViewById(R.id.status);
        selectVideo = findViewById(R.id.select_video);
        selectVideo.setOnClickListener(view -> {
            Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
            intent.addCategory(Intent.CATEGORY_OPENABLE);
            intent.setType("video/*");
            intent.putExtra(Intent.EXTRA_LOCAL_ONLY, true);
            startActivityForResult(intent, PICK_VIDEO);
        });
        if (Intent.ACTION_VIEW.equals(getIntent().getAction()) && getIntent().getData() != null) {
            analyzeUri(getIntent().getData());
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != PICK_VIDEO || resultCode != RESULT_OK || data == null) return;
        Uri video = data.getData();
        if (video == null) return;
        analyzeUri(video);
    }

    private void analyzeUri(Uri video) {
        String scheme = video.getScheme();
        if (!"content".equals(scheme) && !"file".equals(scheme)) {
            status.setText("기기에 저장된 영상만 선택하세요.");
            return;
        }
        selectVideo.setEnabled(false);
        status.setText("휴대폰에서 분석 중입니다. 잠시 기다려 주세요…");
        executor.execute(() -> {
            try {
                PoseAnalyzer.Outcome outcome = PoseAnalyzer.analyze(this, video);
                String verdict = outcome.probability >= outcome.threshold ? "낙상/쓰러짐 가능성 있음" : "낙상 징후 낮음";
                String message = String.format(Locale.KOREAN,
                        "%s\n예측 확률: %.1f%%\n관절 발견: %d/12프레임\n영상 길이: %.1f초\n\n연구용 판정이며 긴 영상이나 실시간 경보 성능은 검증되지 않았습니다.",
                        verdict, outcome.probability * 100, outcome.foundFrames, outcome.durationMs / 1000.0);
                runOnUiThread(() -> {
                    status.setText(message);
                    selectVideo.setEnabled(true);
                });
            } catch (Exception error) {
                runOnUiThread(() -> {
                    status.setText("분석 실패: " + error.getMessage());
                    selectVideo.setEnabled(true);
                });
            }
        });
    }

    @Override
    protected void onDestroy() {
        executor.shutdownNow();
        super.onDestroy();
    }
}
