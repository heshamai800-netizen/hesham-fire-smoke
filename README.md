# HESHAM Fire & Smoke AI — CPU Lite 1.5

Windows 10/11, Python 3.11 or 3.12 (64-bit). No ONNX, CUDA or GPU delegates.

1. Extract ZIP into a new folder. Do NOT merge with a previous `.venv`.
2. Run `install_windows.bat` (internet required for pip packages).
3. Run `run_windows.bat` and open http://127.0.0.1:8765.
4. Add a USB or RTSP camera from the web dashboard.

## Changes
- One TFLite inference per frame produces both Fire and Smoke detections; older builds ran the same model twice.
- Camera capture, AI and 15 FPS capped MJPEG preview run in separate threads; stale frames are skipped.
- No GPU conversion or GPU DLL dependencies. Minimal runtime-only project.
- Choose CPU thread count by editing `HESHAM_CPU_THREADS=4` in `run_windows.bat` (try 2, 4, 6 based on actual measured inference time).
- `AI TIME F/S` reflects the **same shared model pass** in milliseconds, not two separate passes.

## Accuracy / safety
- This is the original third-party TFLite model, unchanged and hash-checked. Manifest lists smoke_visible and flame_visible.
- Original bounding-box decoding and class mapping were retained; these were not independently validated against labeled video. False positives/negatives are possible.
- CPU model inference speed is hardware-dependent; fast video preview does not mean high AI FPS.
- Do not rely on this experimental software as the sole industrial fire-safety system.
