///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2021, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

// ----> Includes
#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "calibration.hpp"
#include "cnpy.h"
#include "stereo.hpp"
// <---- Includes

// INPUT SOURCE
#define USE_LOCAL_VIDEO
// #define USE_GSTREAMER_STREAM
// #define USE_LIVE_ZED_CAMERA

// CALIBRATION SOURCE
// #define USE_SN_CONF_CALIBRATION
#define USE_UNDERWATER_NPY_CALIBRATION

#if (defined(USE_LOCAL_VIDEO) + defined(USE_GSTREAMER_STREAM) + defined(USE_LIVE_ZED_CAMERA)) != 1
#error "Enable exactly one input mode: USE_LOCAL_VIDEO, USE_GSTREAMER_STREAM, or USE_LIVE_ZED_CAMERA"
#endif

#ifdef USE_LIVE_ZED_CAMERA
#include "videocapture.hpp"
#endif

#if defined(USE_SN_CONF_CALIBRATION) && defined(USE_UNDERWATER_NPY_CALIBRATION)
#error "Enable only one calibration mode: USE_SN_CONF_CALIBRATION or USE_UNDERWATER_NPY_CALIBRATION"
#endif

#if !defined(USE_SN_CONF_CALIBRATION) && !defined(USE_UNDERWATER_NPY_CALIBRATION)
#error "Enable one calibration mode: USE_SN_CONF_CALIBRATION or USE_UNDERWATER_NPY_CALIBRATION"
#endif

bool loadUnderwaterCalibration(
    const std::string& calibDir,
    cv::Mat& cameraMatrix_left,
    cv::Mat& cameraMatrix_right,
    cv::Mat& distCoeffs_left,
    cv::Mat& distCoeffs_right,
    cv::Mat& map_left_x,
    cv::Mat& map_left_y,
    cv::Mat& map_right_x,
    cv::Mat& map_right_y,
    double& fx,
    double& fy,
    double& cx,
    double& cy,
    double& baseline)
{
    try
    {
        cnpy::NpyArray kLeftNpy = cnpy::npy_load(calibDir + "/K_left.npy");
        cnpy::NpyArray kRightNpy = cnpy::npy_load(calibDir + "/K_right.npy");
        if (kLeftNpy.shape.size() != 2 || kRightNpy.shape.size() != 2 ||
            kLeftNpy.shape[0] != 3 || kLeftNpy.shape[1] != 3 ||
            kRightNpy.shape[0] != 3 || kRightNpy.shape[1] != 3)
        {
            std::cerr << "Invalid K matrix shapes in " << calibDir << std::endl;
            return false;
        }

        cameraMatrix_left = cv::Mat(3, 3, CV_64F, kLeftNpy.data<double>()).clone();
        cameraMatrix_right = cv::Mat(3, 3, CV_64F, kRightNpy.data<double>()).clone();

        cnpy::NpyArray distLeftNpy = cnpy::npy_load(calibDir + "/dist_left.npy");
        cnpy::NpyArray distRightNpy = cnpy::npy_load(calibDir + "/dist_right.npy");
        const size_t distLeftCount = distLeftNpy.num_vals;
        const size_t distRightCount = distRightNpy.num_vals;
        if (distLeftCount < 5 || distRightCount < 5)
        {
            std::cerr << "Invalid distortion array size in " << calibDir << std::endl;
            return false;
        }

        distCoeffs_left = cv::Mat(1, 5, CV_64F, distLeftNpy.data<double>()).clone();
        distCoeffs_right = cv::Mat(1, 5, CV_64F, distRightNpy.data<double>()).clone();

        cnpy::NpyArray tNpy = cnpy::npy_load(calibDir + "/T.npy");
        if (tNpy.num_vals < 3)
        {
            std::cerr << "Invalid T array size in " << calibDir << std::endl;
            return false;
        }

        cv::Mat T = cv::Mat(3, 1, CV_64F, tNpy.data<double>()).clone();
        baseline = std::abs(T.at<double>(0, 0));
        if (baseline < 1.0)
        {
            baseline *= 1000.0;
        }

        fx = cameraMatrix_left.at<double>(0, 0);
        fy = cameraMatrix_left.at<double>(1, 1);
        cx = cameraMatrix_left.at<double>(0, 2);
        cy = cameraMatrix_left.at<double>(1, 2);

        cnpy::NpyArray lm1 = cnpy::npy_load(calibDir + "/left_map1.npy");
        cnpy::NpyArray lm2 = cnpy::npy_load(calibDir + "/left_map2.npy");
        cnpy::NpyArray rm1 = cnpy::npy_load(calibDir + "/right_map1.npy");
        cnpy::NpyArray rm2 = cnpy::npy_load(calibDir + "/right_map2.npy");

        // Preferred format: CV_32FC1 maps saved as 2D float arrays.
        if (lm1.shape.size() == 2 && lm2.shape.size() == 2 && rm1.shape.size() == 2 && rm2.shape.size() == 2)
        {
            const int rows = static_cast<int>(lm1.shape[0]);
            const int cols = static_cast<int>(lm1.shape[1]);
            const bool sameShape =
                lm2.shape[0] == lm1.shape[0] && lm2.shape[1] == lm1.shape[1] &&
                rm1.shape[0] == lm1.shape[0] && rm1.shape[1] == lm1.shape[1] &&
                rm2.shape[0] == lm1.shape[0] && rm2.shape[1] == lm1.shape[1];
            if (!sameShape)
            {
                std::cerr << "Rectification map sizes do not match" << std::endl;
                return false;
            }

            map_left_x = cv::Mat(rows, cols, CV_32F, lm1.data<float>()).clone();
            map_left_y = cv::Mat(rows, cols, CV_32F, lm2.data<float>()).clone();
            map_right_x = cv::Mat(rows, cols, CV_32F, rm1.data<float>()).clone();
            map_right_y = cv::Mat(rows, cols, CV_32F, rm2.data<float>()).clone();
            return true;
        }

        // Compatibility format: CV_16SC2 + CV_16UC1 map pair as saved by OpenCV.
        if (lm1.shape.size() == 3 && rm1.shape.size() == 3 &&
            lm2.shape.size() == 2 && rm2.shape.size() == 2 &&
            lm1.shape[2] == 2 && rm1.shape[2] == 2)
        {
            const bool sameRowsCols =
                lm2.shape[0] == lm1.shape[0] && lm2.shape[1] == lm1.shape[1] &&
                rm1.shape[0] == lm1.shape[0] && rm1.shape[1] == lm1.shape[1] &&
                rm2.shape[0] == lm1.shape[0] && rm2.shape[1] == lm1.shape[1];
            if (!sameRowsCols)
            {
                std::cerr << "Rectification map sizes do not match" << std::endl;
                return false;
            }

            const int rows = static_cast<int>(lm1.shape[0]);
            const int cols = static_cast<int>(lm1.shape[1]);
            map_left_x = cv::Mat(rows, cols, CV_16SC2, lm1.data<short>()).clone();
            map_left_y = cv::Mat(rows, cols, CV_16UC1, lm2.data<unsigned short>()).clone();
            map_right_x = cv::Mat(rows, cols, CV_16SC2, rm1.data<short>()).clone();
            map_right_y = cv::Mat(rows, cols, CV_16UC1, rm2.data<unsigned short>()).clone();
            return true;
        }

        std::cerr << "Unsupported rectification map layout in " << calibDir << std::endl;
        return false;

    }
    catch (const std::exception& e)
    {
        std::cerr << "Error loading underwater calibration from '" << calibDir << "': " << e.what() << std::endl;
        return false;
    }
}

struct Measurement
{
    cv::Point p1;
    cv::Point p2;
    double distance_cm;
};

enum class AppMode
{
    LIVE_VIEW,
    MEASURE_VIEW
};

struct AppState
{
    AppMode mode = AppMode::LIVE_VIEW;

    cv::Mat frozenImage;
    cv::Mat frozenDepthMap;

    std::vector<Measurement> measurements;
    std::vector<cv::Point> pendingPoints;

    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;

    std::string statusMessage;
};

#ifdef USE_LIVE_ZED_CAMERA
sl_oc::video::VideoCapture* g_zedCapPtr = nullptr;
#else
cv::VideoCapture* g_videoCapPtr = nullptr;
#endif

bool getStereoFrame(cv::Mat& sideBySideFrame)
{
#ifdef USE_LIVE_ZED_CAMERA
    if (!g_zedCapPtr)
        return false;

    const sl_oc::video::Frame frame = g_zedCapPtr->getLastFrame();
    if (!frame.data)
        return false;

    // ZED camera provides frames in YUYV format
    cv::Mat frameYUV(frame.height, frame.width, CV_8UC2, frame.data);
    cv::Mat frameBGR;
    cv::cvtColor(frameYUV, frameBGR, cv::COLOR_YUV2BGR_YUYV);
    sideBySideFrame = frameBGR;
    return true;
#else
    if (!g_videoCapPtr)
        return false;

    cv::Mat frame;
    if (!g_videoCapPtr->read(frame) || frame.empty())
    {
        return false;
    }

    if (frame.channels() == 1)
    {
        cv::cvtColor(frame, sideBySideFrame, cv::COLOR_GRAY2BGR);
    }
    else if (frame.channels() == 4)
    {
        cv::cvtColor(frame, sideBySideFrame, cv::COLOR_BGRA2BGR);
    }
    else
    {
        sideBySideFrame = frame;
    }

    return true;
#endif
}

float sampleDepthNeighborhood(const cv::Mat& depthMap, int x, int y)
{
    const int halfWindow = 3; // 5x5 neighborhood
    std::vector<float> samples;
    samples.reserve(25);

    for (int yy = y - halfWindow; yy <= y + halfWindow; ++yy)
    {
        for (int xx = x - halfWindow; xx <= x + halfWindow; ++xx)
        {
            if (xx < 0 || yy < 0 || xx >= depthMap.cols || yy >= depthMap.rows)
            {
                continue;
            }

            float d = depthMap.at<float>(yy, xx);
            if (std::isfinite(d) && d > 0.0f)
            {
                samples.push_back(d);
            }
        }
    }

    if (samples.empty())
    {
        return std::numeric_limits<float>::quiet_NaN();
    }

    const size_t mid = samples.size() / 2;
    std::nth_element(samples.begin(), samples.begin() + static_cast<long>(mid), samples.end());
    float median = samples[mid];

    if (samples.size() % 2 == 0)
    {
        std::nth_element(samples.begin(), samples.begin() + static_cast<long>(mid - 1), samples.end());
        median = 0.5f * (median + samples[mid - 1]);
    }

    return median;
}

double computeDistanceCm(
    const cv::Point& p1,
    const cv::Point& p2,
    float d1,
    float d2,
    double fx,
    double fy,
    double cx,
    double cy)
{
    const double x1 = (static_cast<double>(p1.x) - cx) * d1 / fx;
    const double y1 = (static_cast<double>(p1.y) - cy) * d1 / fy;
    const double z1 = d1;

    const double x2 = (static_cast<double>(p2.x) - cx) * d2 / fx;
    const double y2 = (static_cast<double>(p2.y) - cy) * d2 / fy;
    const double z2 = d2;

    const double dist_mm = std::sqrt(
        (x1 - x2) * (x1 - x2) +
        (y1 - y2) * (y1 - y2) +
        (z1 - z2) * (z1 - z2));

    return dist_mm / 10.0;
}

std::string formatDistanceLabel(double distanceCm)
{
    std::ostringstream oss;
    oss << std::fixed;
    if (distanceCm < 100.0)
    {
        oss << std::setprecision(1) << distanceCm << " cm";
    }
    else
    {
        oss << std::setprecision(2) << (distanceCm / 100.0) << " m";
    }
    return oss.str();
}

void drawMeasurements(const AppState& state, cv::Mat& display)
{
    for (const auto& m : state.measurements)
    {
        cv::circle(display, m.p1, 4, cv::Scalar(0, 255, 0), -1);
        cv::circle(display, m.p2, 4, cv::Scalar(0, 255, 0), -1);
        cv::line(display, m.p1, m.p2, cv::Scalar(255, 0, 0), 1);

        cv::Point mid((m.p1.x + m.p2.x) / 2, (m.p1.y + m.p2.y) / 2 - 10);
        cv::putText(display,
                    formatDistanceLabel(m.distance_cm),
                    mid,
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.6,
                    cv::Scalar(0, 0, 255),
                    2,
                    cv::LINE_AA);
    }

    if (!state.pendingPoints.empty())
    {
        cv::circle(display, state.pendingPoints.back(), 4, cv::Scalar(0, 255, 255), -1);
    }
}

void drawOverlay(const AppState& state, cv::Mat& display)
{
    const std::string modeLine =
        (state.mode == AppMode::LIVE_VIEW)
            ? "LIVE MODE - press SPACE to capture frame"
            : "MEASUREMENT MODE - click two points";

    const std::string controls = "SPACE: freeze    U: undo    C: clear    R: live    Q: quit";

    cv::putText(display, modeLine, cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(40, 255, 40), 2, cv::LINE_AA);
    cv::putText(display, controls, cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

    if (!state.statusMessage.empty())
    {
        cv::putText(display,
                    state.statusMessage,
                    cv::Point(20, 90),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.6,
                    cv::Scalar(80, 220, 255),
                    2,
                    cv::LINE_AA);
    }
}

void onMouse(int event, int x, int y, int /*flags*/, void* userdata)
{
    if (event != cv::EVENT_LBUTTONDOWN || userdata == nullptr)
    {
        return;
    }

    AppState* state = static_cast<AppState*>(userdata);
    if (state->mode != AppMode::MEASURE_VIEW)
    {
        return;
    }

    if (x < 0 || y < 0 || x >= state->frozenDepthMap.cols || y >= state->frozenDepthMap.rows)
    {
        state->statusMessage = "Click ignored: out of bounds";
        return;
    }

    state->pendingPoints.emplace_back(x, y);
    state->statusMessage.clear();

    if (state->pendingPoints.size() < 2)
    {
        return;
    }

    const cv::Point p1 = state->pendingPoints[0];
    const cv::Point p2 = state->pendingPoints[1];

    const float d1 = sampleDepthNeighborhood(state->frozenDepthMap, p1.x, p1.y);
    const float d2 = sampleDepthNeighborhood(state->frozenDepthMap, p2.x, p2.y);

    if (!std::isfinite(d1) || !std::isfinite(d2) || d1 <= 0.0f || d2 <= 0.0f)
    {
        state->pendingPoints.clear();
        state->statusMessage = "Invalid depth at selected point";
        return;
    }

    const double distanceCm = computeDistanceCm(p1, p2, d1, d2, state->fx, state->fy, state->cx, state->cy);
    state->measurements.push_back({p1, p2, distanceCm});
    state->pendingPoints.clear();

    std::ostringstream oss;
    oss << "Measurement added: " << formatDistanceLabel(distanceCm);
    state->statusMessage = oss.str();
}

bool computeDepthMap(
    const cv::Mat& leftRect,
    const cv::Mat& rightRect,
    cv::Ptr<cv::StereoSGBM>& leftMatcher,
    double fx,
    double baseline,
    cv::Mat& depthMap)
{
    if (leftRect.empty() || rightRect.empty())
    {
        return false;
    }

    cv::Mat leftForMatcher;
    cv::Mat rightForMatcher;
    const double resizeFact = 0.5;

    cv::resize(leftRect, leftForMatcher, cv::Size(), resizeFact, resizeFact, cv::INTER_AREA);
    cv::resize(rightRect, rightForMatcher, cv::Size(), resizeFact, resizeFact, cv::INTER_AREA);

    cv::Mat disparityHalf;
    leftMatcher->compute(leftForMatcher, rightForMatcher, disparityHalf);

    cv::Mat disparity;
    disparityHalf.convertTo(disparity, CV_32FC1);
    disparity *= 1.0f / 16.0f;

    disparity *= 2.0f;
    cv::resize(disparity, disparity, leftRect.size(), 0.0, 0.0, cv::INTER_LINEAR);

    cv::threshold(disparity, disparity, 0.1, 0.0, cv::THRESH_TOZERO);

    cv::Mat filteredDisparity;
    cv::bilateralFilter(disparity, filteredDisparity, 5, 25.0, 25.0);

    const double numerator = fx * baseline;
    cv::divide(numerator, filteredDisparity, depthMap);

    return !depthMap.empty();
}

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

#ifdef USE_LOCAL_VIDEO
    const std::string localVideoPath = "recording.mp4";
    cv::VideoCapture cap(localVideoPath);
    g_videoCapPtr = &cap;
    if (!cap.isOpened())
    {
        std::cerr << "Cannot open input source" << std::endl;
        return EXIT_FAILURE;
    }
#endif

#ifdef USE_GSTREAMER_STREAM
    const std::string gstPipeline =
        "rtspsrc location=rtsp://192.168.1.100:8554/videofeed latency=0 buffer-mode=auto "
        "! decodebin ! videoconvert ! appsink max-buffers=1 drop=True";
    cv::VideoCapture cap(gstPipeline, cv::CAP_GSTREAMER);
    g_videoCapPtr = &cap;
    if (!cap.isOpened())
    {
        std::cerr << "Cannot open input source" << std::endl;
        return EXIT_FAILURE;
    }
#endif

#ifdef USE_LIVE_ZED_CAMERA
    sl_oc::video::VideoParams params;
    params.res = sl_oc::video::RESOLUTION::HD720;
    params.fps = sl_oc::video::FPS::FPS_30;
    params.verbose = sl_oc::VERBOSITY::INFO;

    sl_oc::video::VideoCapture zed(params);
    g_zedCapPtr = &zed;

    if (!zed.initializeVideo(-1))
    {
        std::cerr << "Cannot open ZED camera" << std::endl;
        return EXIT_FAILURE;
    }
#endif

    cv::Mat sideBySideFrame;
    if (!getStereoFrame(sideBySideFrame))
    {
        std::cerr << "Cannot read first frame from input source" << std::endl;
        return EXIT_FAILURE;
    }

    if (sideBySideFrame.cols % 2 != 0)
    {
        std::cerr << "Input frame width must be even (side-by-side stereo expected)" << std::endl;
        return EXIT_FAILURE;
    }

    // Verify frame size for live ZED camera
#ifdef USE_LIVE_ZED_CAMERA
    std::cout << "Camera frame size: " << sideBySideFrame.cols << " x " << sideBySideFrame.rows << std::endl;
    if (sideBySideFrame.cols != 2560 || sideBySideFrame.rows != 720)
    {
        std::cerr << "Unexpected camera resolution. Underwater calibration requires 2560x720 side-by-side." << std::endl;
        return EXIT_FAILURE;
    }
#endif

    const int leftWidth = sideBySideFrame.cols / 2;
    const int frameHeight = sideBySideFrame.rows;
    const cv::Size resolution(leftWidth, frameHeight);

    cv::Mat cameraMatrix_left, cameraMatrix_right;
    cv::Mat distCoeffs_left, distCoeffs_right;
    cv::Mat map_left_x, map_left_y;
    cv::Mat map_right_x, map_right_y;

    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    double baseline = 0.0;

#ifdef USE_SN_CONF_CALIBRATION
    #ifdef USE_LIVE_ZED_CAMERA
    // For live ZED camera, only use local calibration file (no download)
    #endif
    const std::string calibration_file = "SN31223474.conf";
    if (!sl_oc::tools::initCalibration(calibration_file,
                                       resolution,
                                       map_left_x,
                                       map_left_y,
                                       map_right_x,
                                       map_right_y,
                                       cameraMatrix_left,
                                       cameraMatrix_right,
                                       &baseline))
    {
        std::cerr << "Failed to load SN.conf calibration" << std::endl;
        return EXIT_FAILURE;
    }

    fx = cameraMatrix_left.at<double>(0, 0);
    fy = cameraMatrix_left.at<double>(1, 1);
    cx = cameraMatrix_left.at<double>(0, 2);
    cy = cameraMatrix_left.at<double>(1, 2);
    baseline = std::abs(baseline);
#endif

#ifdef USE_UNDERWATER_NPY_CALIBRATION
    const std::string calibDir = "underwater_calibration";
    if (!loadUnderwaterCalibration(calibDir,
                                   cameraMatrix_left,
                                   cameraMatrix_right,
                                   distCoeffs_left,
                                   distCoeffs_right,
                                   map_left_x,
                                   map_left_y,
                                   map_right_x,
                                   map_right_y,
                                   fx,
                                   fy,
                                   cx,
                                   cy,
                                   baseline))
    {
        std::cerr << "Failed to load underwater .npy calibration" << std::endl;
        return EXIT_FAILURE;
    }
#endif

    AppState state;
    state.fx = fx;
    state.fy = fy;
    state.cx = cx;
    state.cy = cy;
    state.statusMessage = "Live mode ready";

    std::cout << "Camera Matrix L:\n" << cameraMatrix_left << std::endl;
    std::cout << "Camera Matrix R:\n" << cameraMatrix_right << std::endl;
    std::cout << "fx: " << fx << std::endl;
    std::cout << "fy: " << fy << std::endl;
    std::cout << "cx: " << cx << std::endl;
    std::cout << "cy: " << cy << std::endl;
    std::cout << "baseline: " << baseline << std::endl;
    std::cout << "Baseline: " << baseline << " mm" << std::endl;

    sl_oc::tools::StereoSgbmPar stereoPar;
    if (!stereoPar.load())
    {
        stereoPar.save();
    }

    cv::Ptr<cv::StereoSGBM> leftMatcher = cv::StereoSGBM::create(
        stereoPar.minDisparity,
        stereoPar.numDisparities,
        stereoPar.blockSize);
    leftMatcher->setMinDisparity(stereoPar.minDisparity);
    leftMatcher->setNumDisparities(stereoPar.numDisparities);
    leftMatcher->setBlockSize(stereoPar.blockSize);
    leftMatcher->setP1(stereoPar.P1);
    leftMatcher->setP2(stereoPar.P2);
    leftMatcher->setDisp12MaxDiff(stereoPar.disp12MaxDiff);
    leftMatcher->setMode(stereoPar.mode);
    leftMatcher->setPreFilterCap(stereoPar.preFilterCap);
    leftMatcher->setUniquenessRatio(stereoPar.uniquenessRatio);
    leftMatcher->setSpeckleWindowSize(stereoPar.speckleWindowSize);
    leftMatcher->setSpeckleRange(stereoPar.speckleRange);
    stereoPar.print();

    cv::namedWindow("Stereo Distance", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("Stereo Distance", onMouse, &state);

    cv::Mat leftRaw;
    cv::Mat rightRaw;
    cv::Mat leftRect;
    cv::Mat rightRect;
    cv::Mat depthMap;

    bool useBufferedFrame = true;
    bool running = true;

    while (running)
    {
        if (state.mode == AppMode::LIVE_VIEW)
        {
            if (!useBufferedFrame)
            {
                if (!getStereoFrame(sideBySideFrame))
                {
                    state.statusMessage = "Input ended or frame read failed";
                    running = false;
                    continue;
                }
            }
            useBufferedFrame = false;

            if (sideBySideFrame.cols % 2 != 0)
            {
                state.statusMessage = "Invalid side-by-side frame width";
                running = false;
                continue;
            }

            leftRaw = sideBySideFrame(cv::Rect(0, 0, sideBySideFrame.cols / 2, sideBySideFrame.rows));
            rightRaw = sideBySideFrame(cv::Rect(sideBySideFrame.cols / 2, 0, sideBySideFrame.cols / 2, sideBySideFrame.rows));

            cv::remap(leftRaw, leftRect, map_left_x, map_left_y, cv::INTER_AREA);
            cv::remap(rightRaw, rightRect, map_right_x, map_right_y, cv::INTER_AREA);

            if (!computeDepthMap(leftRect, rightRect, leftMatcher, state.fx, baseline, depthMap))
            {
                state.statusMessage = "Depth computation failed";
            }

            cv::Mat liveDisplay = leftRect.clone();
            drawOverlay(state, liveDisplay);
            cv::imshow("Stereo Distance", liveDisplay);
        }
        else
        {
            cv::Mat measureDisplay = state.frozenImage.clone();
            drawMeasurements(state, measureDisplay);
            drawOverlay(state, measureDisplay);
            cv::imshow("Stereo Distance", measureDisplay);
        }

        const int key = cv::waitKey(5);
        if (key == 'q' || key == 'Q')
        {
            running = false;
        }
        else if (key == ' ')
        {
            if (state.mode == AppMode::LIVE_VIEW && !leftRect.empty() && !depthMap.empty())
            {
                state.frozenImage = leftRect.clone();
                state.frozenDepthMap = depthMap.clone();
                state.measurements.clear();
                state.pendingPoints.clear();
                state.mode = AppMode::MEASURE_VIEW;
                state.statusMessage = "Frame frozen. Click two points to measure";
            }
        }
        else if (key == 'u' || key == 'U')
        {
            if (state.mode == AppMode::MEASURE_VIEW)
            {
                if (!state.pendingPoints.empty())
                {
                    state.pendingPoints.pop_back();
                    state.statusMessage = "Pending point removed";
                }
                else if (!state.measurements.empty())
                {
                    state.measurements.pop_back();
                    state.statusMessage = "Last measurement removed";
                }
            }
        }
        else if (key == 'c' || key == 'C')
        {
            if (state.mode == AppMode::MEASURE_VIEW)
            {
                state.measurements.clear();
                state.pendingPoints.clear();
                state.statusMessage = "All measurements cleared";
            }
        }
        else if (key == 'r' || key == 'R')
        {
            state.mode = AppMode::LIVE_VIEW;
            state.measurements.clear();
            state.pendingPoints.clear();
            state.frozenImage.release();
            state.frozenDepthMap.release();
            state.statusMessage = "Returned to live mode";
        }
    }

#ifdef USE_LIVE_ZED_CAMERA
    // ZED camera cleanup is handled automatically
#else
    cap.release();
#endif
    cv::destroyAllWindows();
    return EXIT_SUCCESS;
}
