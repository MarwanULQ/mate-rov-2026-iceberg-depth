# Iceberg Depth

Stereo distance measurement tool built on top of the Stereolabs Open Capture API.

The application supports:

* prerecorded side-by-side stereo video files (`.mp4`)
* RTSP / GStreamer stereo streams
* freezing a frame and measuring the real-world distance between multiple point pairs
* interactive undo / clear / reset controls

![useless alt text](/test/3.png)

---

## Installation

First make the scripts executable:

```bash
chmod +x install_prereqs.sh
chmod +x build.sh
chmod +x run_receiver.sh
```

Then install all required dependencies:

```bash
./install_prereqs.sh
```

---

## Build

Build the project with:

```bash
./build.sh
```

The build script configures CMake and compiles the measurement executable.

---

## Run

Run the receiver with:

```bash
./run_receiver.sh
```

---

## Input Source Selection

The application can operate in one of two modes:

1. Local prerecorded stereo video
2. RTSP / GStreamer stereo stream

The mode is selected in:

```text
src/iceberg_depth/examples/measure_distance_v1.cpp
```

using compile-time `#define` directives.

### Local Video Mode

Enable:

```cpp
#define USE_LOCAL_VIDEO
// #define USE_GSTREAMER_STREAM
```

When enabled, the program loads a local side-by-side stereo recording:

```cpp
const std::string localVideoPath = "recording.mp4";
```

Requirements:

* the video must be side-by-side stereo
* the video path must match the location of the file (put the video in the repo's root and rename it to 'recording')
* [prerecorded videos can be found here](https://drive.google.com/drive/u/1/folders/1ywYO08Prxha8mJFXOx24Lt_mlt3WUB-b)

---

### RTSP / GStreamer Mode

Enable:

```cpp
// #define USE_LOCAL_VIDEO
#define USE_GSTREAMER_STREAM
```

Then edit the RTSP pipeline in the same source file:

```cpp
const std::string pipeline =
    "rtspsrc location=rtsp://<ip>:<port>/videofeed ! decodebin ! videoconvert ! appsink";
```

After changing the define or RTSP URL, rebuild the project:

```bash
./build.sh
```

---

## Calibration File

The file:

```text
SN31223474.conf
```

---

## Stereo Parameter Tuning

The repository includes a StereoSGBM tuning tool that provides a gui to tune the disparity map generated:

```text
src/iceberg_depth/examples/tools/zed_oc_tune_stereo_sgbm.cpp
```

Run the tuning executable from the build directory:

```bash
cd src/iceberg_depth/build
./zed_open_capture_depth_tune_stereo
```

Adjust the StereoSGBM parameters, then save them.

The measurement application automatically loads the saved stereo parameters file on startup.

It works by capturing the first frame and providing a gui over it with controls. (can't be used with local video because it works on the first frame captured)

---

## Measurement Workflow

1. Start the application
2. Wait until the live video appears
3. Press `SPACE` to freeze the current frame
4. Click two points to create one measurement
5. Repeat to create additional measurements

Each pair of points is connected by a thin line and the measured distance is displayed above the line.

---

## Controls

| Key              | Action                             |
| ---------------- | ---------------------------------- |
| `SPACE`          | Freeze the current frame           |
| Left click twice | Create one distance measurement    |
| `U`              | Undo the last point or measurement |
| `C`              | Clear all measurements             |
| `R`              | Return to live mode                |
| `Q`              | Quit the application               |

---

## Notes

* The application does not generate a point cloud.
* Distance is computed directly from the stereo depth map.
* Measurements are more stable when StereoSGBM has been tuned first.

---
