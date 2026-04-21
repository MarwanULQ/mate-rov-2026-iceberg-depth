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
#include <array>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>

#include "calibration.hpp"
#include "cnpy.h"
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

#ifndef DEFAULT_MODEL_CONFIG_PATH
#define DEFAULT_MODEL_CONFIG_PATH "exports/stereoanywhere2_torchscript.pt"
#endif

#ifndef DEFAULT_LOCAL_VIDEO_PATH
#define DEFAULT_LOCAL_VIDEO_PATH "recording.mp4"
#endif

#ifndef DEFAULT_SN_CALIBRATION_PATH
#define DEFAULT_SN_CALIBRATION_PATH "SN31223474.conf"
#endif

#ifndef DEFAULT_GSTREAMER_PIPELINE
#define DEFAULT_GSTREAMER_PIPELINE "rtspsrc location=rtsp://192.168.1.100:8554/videofeed latency=0 buffer-mode=auto ! decodebin ! videoconvert ! appsink max-buffers=1 drop=True"
#endif

#ifndef DEFAULT_UNDERWATER_CALIB_DIR
#define DEFAULT_UNDERWATER_CALIB_DIR "underwater_calibration"
#endif

#ifndef DEFAULT_MODEL_INPUT_HEIGHT
#define DEFAULT_MODEL_INPUT_HEIGHT 720
#endif

#ifndef DEFAULT_MODEL_INPUT_WIDTH
#define DEFAULT_MODEL_INPUT_WIDTH 1280
#endif

#ifndef APP_WINDOW_NAME
#define APP_WINDOW_NAME "Stereo Distance v4"
#endif

constexpr const char* kDefaultModelConfigPath = DEFAULT_MODEL_CONFIG_PATH;

std::string resolvePathFromEnvOrDefault(const char* envKey, const char* fallback)
{
    const char* envValue = std::getenv(envKey);
    if (envValue && envValue[0] != '\0')
    {
        return std::string(envValue);
    }

    return std::string(fallback);
}

double normalizeBaselineMm(double baseline)
{
    baseline = std::abs(baseline);
    if (baseline < 1.0)
    {
        baseline *= 1000.0;
    }

    return baseline;
}

std::string requestedTorchDeviceName()
{
    const char* envValue = std::getenv("STEREOANYWHERE_TORCH_DEVICE");
    if (!envValue || envValue[0] == '\0')
    {
        return "cpu";
    }

    std::string value(envValue);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    if (value == "cuda" || value == "gpu")
    {
        return "cuda";
    }

    return "cpu";
}

struct TorchRuntimeConfig
{
    std::string configPath;
    std::string modelPath;
    std::string outputKey = "refdisp";
    int inputHeight = DEFAULT_MODEL_INPUT_HEIGHT;
    int inputWidth = DEFAULT_MODEL_INPUT_WIDTH;
};

bool loadTorchRuntimeConfig(const std::string& modelSpecPath, TorchRuntimeConfig& cfg)
{
    try
    {
        const std::filesystem::path specPath(modelSpecPath);
        std::string ext = specPath.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });

        if (ext == ".pt" || ext == ".pth")
        {
            cfg.configPath.clear();
            cfg.modelPath = modelSpecPath;

            if (!std::filesystem::exists(cfg.modelPath))
            {
                std::cerr << "TorchScript model not found: " << cfg.modelPath << std::endl;
                return false;
            }

            if (cfg.inputHeight <= 0 || cfg.inputWidth <= 0)
            {
                std::cerr << "Invalid default model input size" << std::endl;
                return false;
            }

            return true;
        }

        std::ifstream in(modelSpecPath);
        if (!in.is_open())
        {
            std::cerr << "Cannot open model config: " << modelSpecPath << std::endl;
            return false;
        }

        nlohmann::json root;
        in >> root;

        cfg.configPath = modelSpecPath;
        if (root.contains("output_key") && root["output_key"].is_string())
        {
            cfg.outputKey = root["output_key"].get<std::string>();
        }

        if (root.contains("pred_size") && root["pred_size"].is_array() && root["pred_size"].size() == 2)
        {
            cfg.inputHeight = root["pred_size"][0].get<int>();
            cfg.inputWidth = root["pred_size"][1].get<int>();
        }

        if (cfg.inputHeight <= 0 || cfg.inputWidth <= 0)
        {
            std::cerr << "Invalid pred_size in model config" << std::endl;
            return false;
        }

        std::filesystem::path modelPath;
        if (root.contains("model_path") && root["model_path"].is_string())
        {
            modelPath = root["model_path"].get<std::string>();
        }
        else if (root.contains("torchscript") && root["torchscript"].is_string())
        {
            modelPath = root["torchscript"].get<std::string>();
        }
        else
        {
            modelPath = specPath.parent_path() / (specPath.stem().string() + ".pt");
        }

        cfg.modelPath = modelPath.string();
        if (!std::filesystem::exists(cfg.modelPath))
        {
            std::cerr << "TorchScript model not found: " << cfg.modelPath << std::endl;
            return false;
        }

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to parse model config '" << modelSpecPath << "': " << e.what() << std::endl;
        return false;
    }
}

class TorchDisparityInference
{
public:
    bool initialize(const TorchRuntimeConfig& cfg)
    {
        try
        {
            inputHeight_ = cfg.inputHeight;
            inputWidth_ = cfg.inputWidth;
            outputKey_ = cfg.outputKey;

            if (inputHeight_ <= 0 || inputWidth_ <= 0)
            {
                lastError_ = "Invalid model input size";
                return false;
            }

            const bool requestCuda = requestedTorchDeviceName() == "cuda";
            const std::vector<torch::Device> candidates = requestCuda
                                                              ? std::vector<torch::Device>{torch::Device(torch::kCUDA), torch::Device(torch::kCPU)}
                                                              : std::vector<torch::Device>{torch::Device(torch::kCPU)};

            for (const torch::Device& candidate : candidates)
            {
                try
                {
                    if (candidate.is_cuda() && !torch::cuda::is_available())
                    {
                        continue;
                    }

                    module_ = torch::jit::load(cfg.modelPath, candidate);
                    module_.eval();
                    device_ = candidate;
                    backendName_ = candidate.is_cuda() ? "CUDA" : "CPU";
                    lastError_.clear();
                    return true;
                }
                catch (const std::exception& e)
                {
                    lastError_ = e.what();
                    if (candidate.is_cuda())
                    {
                        std::cerr << "CUDA TorchScript load failed, falling back to CPU: " << e.what() << std::endl;
                    }
                }
            }

            if (lastError_.empty())
            {
                lastError_ = "TorchScript model could not be initialized";
            }

            return false;
        }
        catch (const std::exception& e)
        {
            lastError_ = e.what();
            std::cerr << "Failed to initialize Torch model: " << e.what() << std::endl;
            std::cerr << "Hint: this app requires a TorchScript model loadable via torch::jit::load." << std::endl;
            std::cerr << "If your .pt was saved with torch.save(model), re-export with torch.jit.trace or torch.jit.script." << std::endl;
            return false;
        }
    }

    bool infer(const cv::Mat& leftRect, const cv::Mat& rightRect, cv::Mat& disparityOut)
    {
        if (leftRect.empty() || rightRect.empty())
        {
            lastError_ = "Empty rectified input frame";
            return false;
        }

        try
        {
            const torch::NoGradGuard noGrad;

            const torch::Tensor leftTensor = makeInputTensor(leftRect);
            const torch::Tensor rightTensor = makeInputTensor(rightRect);

            std::vector<torch::jit::IValue> inputs;
            inputs.emplace_back(leftTensor);
            inputs.emplace_back(rightTensor);

            const torch::jit::IValue output = module_.forward(inputs);

            torch::Tensor dispTensor;
            if (!extractOutputTensor(output, dispTensor))
            {
                lastError_ = "Unsupported model output format";
                std::cerr << lastError_ << std::endl;
                return false;
            }

            dispTensor = dispTensor.squeeze().detach().to(torch::kFloat32).to(torch::kCPU).contiguous();

            if (dispTensor.dim() != 2)
            {
                std::ostringstream oss;
                oss << "Unexpected disparity tensor dimensions: " << dispTensor.sizes();
                lastError_ = oss.str();
                std::cerr << lastError_ << std::endl;
                return false;
            }

            const int modelOutH = static_cast<int>(dispTensor.size(0));
            const int modelOutW = static_cast<int>(dispTensor.size(1));

            cv::Mat disparityModel(modelOutH, modelOutW, CV_32FC1, dispTensor.data_ptr<float>());
            cv::Mat disparityResized;
            cv::resize(disparityModel, disparityResized, leftRect.size(), 0.0, 0.0, cv::INTER_LINEAR);

            const float widthScale = static_cast<float>(leftRect.cols) / static_cast<float>(modelOutW);
            disparityOut = disparityResized.clone();
            disparityOut *= widthScale;

            cv::threshold(disparityOut, disparityOut, 0.01, 0.0, cv::THRESH_TOZERO);
            lastError_.clear();
            return true;
        }
        catch (const std::exception& e)
        {
            lastError_ = e.what();
            std::cerr << "Torch inference failed: " << e.what() << std::endl;
            return false;
        }
    }

    std::string deviceName() const
    {
        return backendName_;
    }

    const std::string& lastError() const
    {
        return lastError_;
    }

private:
    torch::Tensor makeInputTensor(const cv::Mat& src) const
    {
        cv::Mat resized;
        cv::resize(src, resized, cv::Size(inputWidth_, inputHeight_), 0.0, 0.0, cv::INTER_AREA);

        cv::Mat rgb;
        if (resized.channels() == 1)
        {
            cv::cvtColor(resized, rgb, cv::COLOR_GRAY2RGB);
        }
        else if (resized.channels() == 4)
        {
            cv::cvtColor(resized, rgb, cv::COLOR_BGRA2RGB);
        }
        else
        {
            cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        }

        cv::Mat rgbFloat;
        rgb.convertTo(rgbFloat, CV_32FC3, 1.0 / 255.0);

        auto tensor = torch::from_blob(
            rgbFloat.data,
            {1, inputHeight_, inputWidth_, 3},
            torch::TensorOptions().dtype(torch::kFloat32));

        tensor = tensor.permute({0, 3, 1, 2}).contiguous().clone();
        return tensor.to(device_);
    }

    bool extractOutputTensor(const torch::jit::IValue& output, torch::Tensor& outTensor) const
    {
        if (output.isTensor())
        {
            outTensor = output.toTensor();
            return true;
        }

        if (output.isTuple())
        {
            const auto tuplePtr = output.toTuple();
            if (!tuplePtr || tuplePtr->elements().empty() || !tuplePtr->elements()[0].isTensor())
            {
                return false;
            }

            outTensor = tuplePtr->elements()[0].toTensor();
            return true;
        }

        if (output.isGenericDict())
        {
            const c10::impl::GenericDict dict = output.toGenericDict();
            for (const auto& item : dict)
            {
                if (item.key().isString() && item.key().toStringRef() == outputKey_ && item.value().isTensor())
                {
                    outTensor = item.value().toTensor();
                    return true;
                }
            }

            for (const auto& item : dict)
            {
                if (item.value().isTensor())
                {
                    outTensor = item.value().toTensor();
                    return true;
                }
            }
        }

        return false;
    }

private:
    torch::jit::script::Module module_;
    torch::Device device_ = torch::kCPU;
    int inputHeight_ = 616;
    int inputWidth_ = 1218;
    std::string outputKey_ = "refdisp";
    std::string backendName_ = "CPU";
    std::string lastError_;
};

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
        baseline = normalizeBaselineMm(T.at<double>(0, 0));

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
    cv::Mat frozenRawDisparityMap;
    cv::Mat frozenFilteredDisparityMap;
    cv::Mat frozenValidityMask;
    cv::Mat frozenDisparityMap;
    cv::Mat frozenDepthMap;

    std::vector<Measurement> measurements;
    std::vector<cv::Point> pendingPoints;

    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;

    bool showRawDisparity = false;
    bool showFilteredDisparity = true;
    bool showValidityMask = false;
    bool showDepthMap = false;

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
    // Collect valid depth samples in a neighborhood around (x, y)
    // 1	3×3
    // 2	5×5
    // 3	7×7
    // 4	9×9
    const int halfWindow = 3;
    std::vector<float> samples;
    samples.reserve(49);

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

cv::Mat makeDisparityDisplay(const cv::Mat& disparity)
{
    if (disparity.empty())
    {
        return cv::Mat();
    }

    cv::Mat positiveDisparity;
    cv::threshold(disparity, positiveDisparity, 0.01, 0.0, cv::THRESH_TOZERO);

    cv::Mat normalized;
    cv::normalize(positiveDisparity, normalized, 0, 255, cv::NORM_MINMAX);

    cv::Mat disparity8u;
    normalized.convertTo(disparity8u, CV_8U);

    return disparity8u;
}

cv::Mat makeDepthDisplay(const cv::Mat& depthMap)
{
    if (depthMap.empty())
    {
        return cv::Mat();
    }

    cv::Mat validMask = (depthMap > 0.0f) & (depthMap < 10000.0f);
    if (cv::countNonZero(validMask) == 0)
    {
        return cv::Mat();
    }

    cv::Mat clipped = depthMap.clone();
    clipped.setTo(0.0f, ~validMask);

    double minVal = 0.0;
    double maxVal = 0.0;
    cv::minMaxLoc(clipped, &minVal, &maxVal, nullptr, nullptr, validMask);
    if (maxVal <= minVal)
    {
        return cv::Mat();
    }

    cv::Mat normalized;
    cv::normalize(clipped, normalized, 0, 255, cv::NORM_MINMAX, CV_8U, validMask);

    cv::Mat color;
    cv::applyColorMap(normalized, color, cv::COLORMAP_TURBO);
    color.setTo(cv::Scalar(0, 0, 0), ~validMask);
    return color;
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

    const std::string controls = "SPACE: freeze  U: undo  C: clear  R: live  D/F/M/Z: debug  Q: quit";

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
    TorchDisparityInference& inferencer,
    double fx,
    double baseline,
    cv::Mat& rawDisparityMap,
    cv::Mat& filteredDisparityMap,
    cv::Mat& validityMask,
    cv::Mat& disparityMap,
    cv::Mat& depthMap)
{
    if (leftRect.empty() || rightRect.empty())
    {
        return false;
    }

    if (!inferencer.infer(leftRect, rightRect, disparityMap))
    {
        return false;
    }

    rawDisparityMap = disparityMap.clone();
    filteredDisparityMap = disparityMap.clone();
    validityMask = (filteredDisparityMap > 0.0f);
    validityMask.convertTo(validityMask, CV_8U, 255.0);

    const double numerator = fx * baseline;
    cv::divide(numerator, filteredDisparityMap, depthMap);
    cv::patchNaNs(depthMap, 0.0);

    if (!validityMask.empty())
    {
        depthMap.setTo(0.0f, validityMask == 0);
    }

    return !depthMap.empty();
}

int main(int argc, char* argv[])
{
    const std::string defaultModelSpec = resolvePathFromEnvOrDefault("STEREOANYWHERE_MODEL_SPEC", kDefaultModelConfigPath);
    std::string modelSpecPath = defaultModelSpec;
    if (argc > 1)
    {
        modelSpecPath = argv[1];
    }

#ifdef USE_LOCAL_VIDEO
    const std::string localVideoPath = resolvePathFromEnvOrDefault("STEREOANYWHERE_INPUT_VIDEO", DEFAULT_LOCAL_VIDEO_PATH);
    cv::VideoCapture cap(localVideoPath);
    g_videoCapPtr = &cap;
    if (!cap.isOpened())
    {
        std::cerr << "Cannot open input source" << std::endl;
        return EXIT_FAILURE;
    }
#endif

#ifdef USE_GSTREAMER_STREAM
    const std::string gstPipeline = resolvePathFromEnvOrDefault("STEREOANYWHERE_GST_PIPELINE", DEFAULT_GSTREAMER_PIPELINE);
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
    const std::string calibration_file = resolvePathFromEnvOrDefault("STEREOANYWHERE_SN_CALIB", DEFAULT_SN_CALIBRATION_PATH);
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
    baseline = normalizeBaselineMm(baseline);
#endif

#ifdef USE_UNDERWATER_NPY_CALIBRATION
    const std::string calibDir = resolvePathFromEnvOrDefault("STEREOANYWHERE_UNDERWATER_CALIB_DIR", DEFAULT_UNDERWATER_CALIB_DIR);
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
    state.statusMessage = "Loading TorchScript model";

    std::cout << "Camera Matrix L:\n" << cameraMatrix_left << std::endl;
    std::cout << "Camera Matrix R:\n" << cameraMatrix_right << std::endl;
    std::cout << "fx: " << fx << std::endl;
    std::cout << "fy: " << fy << std::endl;
    std::cout << "cx: " << cx << std::endl;
    std::cout << "cy: " << cy << std::endl;
    std::cout << "baseline: " << baseline << std::endl;
    std::cout << "Baseline: " << baseline << " mm" << std::endl;

    TorchRuntimeConfig runtimeCfg;
    if (!loadTorchRuntimeConfig(modelSpecPath, runtimeCfg))
    {
        return EXIT_FAILURE;
    }

    TorchDisparityInference inferencer;
    if (!inferencer.initialize(runtimeCfg))
    {
        std::vector<std::string> fallbackModelSpecs;
        const std::filesystem::path requestedSpec(modelSpecPath);
        const std::filesystem::path requestedDir = requestedSpec.has_parent_path() ? requestedSpec.parent_path() : std::filesystem::path(".");

        fallbackModelSpecs.emplace_back((requestedDir / "stereoanywhere_torchscript.pt").string());
        fallbackModelSpecs.emplace_back((requestedDir / "stereoadapter_refdisp.pt").string());
        fallbackModelSpecs.emplace_back((requestedDir / "stereoadapter_refdisp.json").string());

        bool fallbackLoaded = false;
        for (const std::string& fallbackSpec : fallbackModelSpecs)
        {
            if (fallbackSpec == modelSpecPath)
            {
                continue;
            }

            TorchRuntimeConfig fallbackCfg;
            if (!loadTorchRuntimeConfig(fallbackSpec, fallbackCfg))
            {
                continue;
            }

            std::cout << "Trying fallback model spec: " << fallbackSpec << std::endl;
            if (inferencer.initialize(fallbackCfg))
            {
                runtimeCfg = fallbackCfg;
                fallbackLoaded = true;
                break;
            }
        }

        if (!fallbackLoaded)
        {
            std::cerr << "Torch backend could not be initialized: " << inferencer.lastError() << std::endl;
            return EXIT_FAILURE;
        }
    }

    std::cout << "Model config: " << (runtimeCfg.configPath.empty() ? "<direct .pt path>" : runtimeCfg.configPath) << std::endl;
    std::cout << "Model path: " << runtimeCfg.modelPath << std::endl;
    std::cout << "Model input size: " << runtimeCfg.inputWidth << "x" << runtimeCfg.inputHeight << std::endl;
    std::cout << "Model output key: " << runtimeCfg.outputKey << std::endl;
    std::cout << "Inference device: " << inferencer.deviceName() << std::endl;

    {
        std::ostringstream oss;
        oss << "Ready - " << inferencer.deviceName() << " backend, model "
            << runtimeCfg.inputWidth << "x" << runtimeCfg.inputHeight;
        state.statusMessage = oss.str();
    }

    cv::namedWindow(APP_WINDOW_NAME, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(APP_WINDOW_NAME, onMouse, &state);

    cv::Mat leftRaw;
    cv::Mat rightRaw;
    cv::Mat leftRect;
    cv::Mat rightRect;
    cv::Mat rawDisparityMap;
    cv::Mat filteredDisparityMap;
    cv::Mat validityMask;

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
                    state.statusMessage = "Input frame skipped: read failed";
                    continue;
                }
            }
            useBufferedFrame = false;

            if (sideBySideFrame.cols % 2 != 0)
            {
                state.statusMessage = "Invalid side-by-side frame width";
                continue;
            }

            leftRaw = sideBySideFrame(cv::Rect(0, 0, sideBySideFrame.cols / 2, sideBySideFrame.rows));
            rightRaw = sideBySideFrame(cv::Rect(sideBySideFrame.cols / 2, 0, sideBySideFrame.cols / 2, sideBySideFrame.rows));

            cv::remap(leftRaw, leftRect, map_left_x, map_left_y, cv::INTER_AREA);
            cv::remap(rightRaw, rightRect, map_right_x, map_right_y, cv::INTER_AREA);

            cv::Mat liveDisplay = leftRect.clone();
            drawOverlay(state, liveDisplay);
            cv::imshow(APP_WINDOW_NAME, liveDisplay);

            // Keep live mode light: no model inference and no debug maps.
            cv::destroyWindow("Raw Disparity");
            cv::destroyWindow("Filtered Disparity");
            cv::destroyWindow("LR Validity Mask");
            cv::destroyWindow("Depth Map");
        }
        else
        {
            cv::Mat measureDisplay = state.frozenImage.clone();
            drawMeasurements(state, measureDisplay);
            drawOverlay(state, measureDisplay);
            cv::imshow(APP_WINDOW_NAME, measureDisplay);

            if (state.showRawDisparity)
            {
                cv::Mat rawDispDisplay = makeDisparityDisplay(state.frozenRawDisparityMap);
                if (!rawDispDisplay.empty())
                {
                    cv::imshow("Raw Disparity", rawDispDisplay);
                }
            }
            else
            {
                cv::destroyWindow("Raw Disparity");
            }

            if (state.showFilteredDisparity)
            {
                cv::Mat filteredDispDisplay = makeDisparityDisplay(state.frozenFilteredDisparityMap);
                if (!filteredDispDisplay.empty())
                {
                    cv::imshow("Filtered Disparity", filteredDispDisplay);
                }
            }
            else
            {
                cv::destroyWindow("Filtered Disparity");
            }

            if (state.showValidityMask)
            {
                if (!state.frozenValidityMask.empty())
                {
                    cv::imshow("LR Validity Mask", state.frozenValidityMask);
                }
            }
            else
            {
                cv::destroyWindow("LR Validity Mask");
            }

            if (state.showDepthMap)
            {
                cv::Mat depthDisplay = makeDepthDisplay(state.frozenDepthMap);
                if (!depthDisplay.empty())
                {
                    cv::imshow("Depth Map", depthDisplay);
                }
            }
            else
            {
                cv::destroyWindow("Depth Map");
            }
        }

        const int key = cv::waitKey(5);
        if (key == 'q' || key == 'Q')
        {
            running = false;
        }
        else if (key == ' ')
        {
            if (state.mode == AppMode::LIVE_VIEW && !leftRect.empty() && !rightRect.empty())
            {
                const auto computeStart = std::chrono::steady_clock::now();
                cv::Mat frozenDisparityMap;
                cv::Mat frozenRawDisparityMap;
                cv::Mat frozenFilteredDisparityMap;
                cv::Mat frozenValidityMask;
                cv::Mat frozenDepthMap;
                if (!computeDepthMap(leftRect,
                                     rightRect,
                                     inferencer,
                                     state.fx,
                                     baseline,
                                     frozenRawDisparityMap,
                                     frozenFilteredDisparityMap,
                                     frozenValidityMask,
                                     frozenDisparityMap,
                                     frozenDepthMap))
                {
                    if (!inferencer.lastError().empty())
                    {
                        state.statusMessage = std::string("Depth computation failed: ") + inferencer.lastError();
                    }
                    else
                    {
                        state.statusMessage = "Depth computation failed";
                    }
                    continue;
                }

                const auto computeEnd = std::chrono::steady_clock::now();
                const double computeMs = std::chrono::duration<double, std::milli>(computeEnd - computeStart).count();

                state.frozenImage = leftRect.clone();
                state.frozenRawDisparityMap = frozenRawDisparityMap;
                state.frozenFilteredDisparityMap = frozenFilteredDisparityMap;
                state.frozenValidityMask = frozenValidityMask;
                state.frozenDisparityMap = frozenDisparityMap;
                state.frozenDepthMap = frozenDepthMap;
                state.measurements.clear();
                state.pendingPoints.clear();
                state.mode = AppMode::MEASURE_VIEW;
                std::ostringstream oss;
                oss << "Frame frozen in " << std::fixed << std::setprecision(1) << computeMs << " ms. Click two points to measure";
                state.statusMessage = oss.str();
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
            state.frozenRawDisparityMap.release();
            state.frozenFilteredDisparityMap.release();
            state.frozenValidityMask.release();
            state.frozenDisparityMap.release();
            state.frozenDepthMap.release();
            state.statusMessage = "Returned to live mode";
        }
        else if (key == 'd' || key == 'D')
        {
            state.showRawDisparity = !state.showRawDisparity;
            state.statusMessage = state.showRawDisparity ? "Raw disparity ON" : "Raw disparity OFF";
        }
        else if (key == 'f' || key == 'F')
        {
            state.showFilteredDisparity = !state.showFilteredDisparity;
            state.statusMessage = state.showFilteredDisparity ? "Filtered disparity ON" : "Filtered disparity OFF";
        }
        else if (key == 'm' || key == 'M')
        {
            state.showValidityMask = !state.showValidityMask;
            state.statusMessage = state.showValidityMask ? "Validity mask ON" : "Validity mask OFF";
        }
        else if (key == 'z' || key == 'Z')
        {
            state.showDepthMap = !state.showDepthMap;
            state.statusMessage = state.showDepthMap ? "Depth debug view ON" : "Depth debug view OFF";
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
