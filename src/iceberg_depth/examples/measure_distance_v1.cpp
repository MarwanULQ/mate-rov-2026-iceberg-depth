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
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "calibration.hpp"
#include "stereo.hpp"
// <---- Includes

#define USE_LOCAL_VIDEO
// #define USE_GSTREAMER_STREAM

#if defined(USE_LOCAL_VIDEO) && defined(USE_GSTREAMER_STREAM)
#error "Enable only one input mode: USE_LOCAL_VIDEO or USE_GSTREAMER_STREAM"
#endif

#if !defined(USE_LOCAL_VIDEO) && !defined(USE_GSTREAMER_STREAM)
#error "Enable one input mode: USE_LOCAL_VIDEO or USE_GSTREAMER_STREAM"
#endif

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

bool getStereoFrame(cv::VideoCapture& cap, cv::Mat& sideBySideFrame)
{
    cv::Mat frame;
    if (!cap.read(frame) || frame.empty())
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

    const std::string calibrationFile = "SN31223474.conf";

#ifdef USE_LOCAL_VIDEO
    const std::string localVideoPath = "recording.mp4";
    cv::VideoCapture cap(localVideoPath);
#endif

#ifdef USE_GSTREAMER_STREAM
    const std::string gstPipeline =
        "rtspsrc location=rtsp://192.168.1.100:8554/videofeed latency=0 buffer-mode=auto "
        "! decodebin ! videoconvert ! appsink max-buffers=1 drop=True";
    cv::VideoCapture cap(gstPipeline, cv::CAP_GSTREAMER);
#endif

    if (!cap.isOpened())
    {
        std::cerr << "Cannot open input source" << std::endl;
        return EXIT_FAILURE;
    }

    cv::Mat sideBySideFrame;
    if (!getStereoFrame(cap, sideBySideFrame))
    {
        std::cerr << "Cannot read first frame from input source" << std::endl;
        return EXIT_FAILURE;
    }

    if (sideBySideFrame.cols % 2 != 0)
    {
        std::cerr << "Input frame width must be even (side-by-side stereo expected)" << std::endl;
        return EXIT_FAILURE;
    }

    const int leftWidth = sideBySideFrame.cols / 2;
    const int frameHeight = sideBySideFrame.rows;

    cv::Mat mapLeftX, mapLeftY, mapRightX, mapRightY;
    cv::Mat cameraMatrixLeft, cameraMatrixRight;
    double baseline = 0.0;
    sl_oc::tools::initCalibration(calibrationFile,
                                  cv::Size(leftWidth, frameHeight),
                                  mapLeftX,
                                  mapLeftY,
                                  mapRightX,
                                  mapRightY,
                                  cameraMatrixLeft,
                                  cameraMatrixRight,
                                  &baseline);

    AppState state;
    state.fx = cameraMatrixLeft.at<double>(0, 0);
    state.fy = cameraMatrixLeft.at<double>(1, 1);
    state.cx = cameraMatrixLeft.at<double>(0, 2);
    state.cy = cameraMatrixLeft.at<double>(1, 2);
    state.statusMessage = "Live mode ready";

    std::cout << "Camera Matrix L:\n" << cameraMatrixLeft << std::endl;
    std::cout << "Camera Matrix R:\n" << cameraMatrixRight << std::endl;
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
                if (!getStereoFrame(cap, sideBySideFrame))
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

            cv::remap(leftRaw, leftRect, mapLeftX, mapLeftY, cv::INTER_AREA);
            cv::remap(rightRaw, rightRect, mapRightX, mapRightY, cv::INTER_AREA);

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

    cap.release();
    cv::destroyAllWindows();
    return EXIT_SUCCESS;
}
