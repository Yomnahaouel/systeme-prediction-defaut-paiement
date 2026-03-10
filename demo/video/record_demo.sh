#!/bin/bash
# Video Demo Recording Script
# Records screen while demonstrating the Credit Risk Prediction System

OUTPUT_DIR="$HOME/systeme-prediction-defaut-paiement/demo/video"
OUTPUT_FILE="$OUTPUT_DIR/demo_recording.mp4"

echo "🎬 Starting Video Demo Recording..."
echo "Recording will capture the full screen"
echo ""

# Start recording in background
ffmpeg -y -video_size 1280x800 -framerate 30 -f x11grab -i :0.0 \
    -c:v libx264 -preset ultrafast -crf 23 \
    "$OUTPUT_FILE" 2>/dev/null &
FFMPEG_PID=$!

echo "Recording started (PID: $FFMPEG_PID)"
sleep 2

# Open Firefox with dashboard
echo "Opening Dashboard..."
firefox http://localhost:8501 &
sleep 5

# Use xdotool to interact
echo "Navigating demo..."

# Wait for page load
sleep 3

# Click on Feature Importance (using keyboard navigation)
xdotool key Tab Tab Tab Down
sleep 3

# Click on Model Info
xdotool key Down
sleep 3

# Open API docs in new tab
firefox http://localhost:8000/docs &
sleep 4

# Show some scrolling
xdotool key Page_Down
sleep 2
xdotool key Page_Down
sleep 2

# Back to terminal for CLI demo
xdotool key alt+Tab
sleep 1

echo ""
echo "Running CLI Demo..."
sleep 2

# Run a quick API test
curl -s http://localhost:8000/ | python3 -m json.tool
sleep 2

curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"EXT_SOURCE_2": 0.6, "AMT_CREDIT": 500000}}' | python3 -m json.tool
sleep 3

echo ""
echo "🎬 Demo Complete! Stopping recording..."

# Stop recording
kill $FFMPEG_PID 2>/dev/null
sleep 2

echo ""
echo "✅ Video saved to: $OUTPUT_FILE"
ls -lh "$OUTPUT_FILE"
