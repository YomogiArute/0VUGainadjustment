#include <iostream>
#include <vector>
#include <cmath>
#include <sndfile.hh>
#include <Eigen/Dense>
#include <algorithm>

// コンスタントの定義
const double VU_REF_DB = -18.0;  // 0 VU を -18 dBFS に設定
const double REF_LEVEL = std::pow(10.0, VU_REF_DB / 20.0);

// ファイルの読み込み
Eigen::MatrixXd readWavFile(const std::string& filePath, int& sampleRate) {
    SndfileHandle fileHandle(filePath);
    if (fileHandle.error()) {
        std::cerr << "Error reading file: " << filePath << std::endl;
        exit(1);
    }
    sampleRate = fileHandle.samplerate();
    int channels = fileHandle.channels();
    int frames = fileHandle.frames();
    Eigen::MatrixXd data(frames, channels);

    std::vector<short> buffer(frames * channels);
    fileHandle.read(buffer.data(), frames * channels);

    for (int i = 0; i < frames; ++i) {
        for (int j = 0; j < channels; ++j) {
            data(i, j) = buffer[i * channels + j] / static_cast<double>(std::numeric_limits<short>::max());
        }
    }
    return data;
}

// VU メーターフィルタ（約 300ms のリリースタイム）
Eigen::VectorXd vuMeter(const Eigen::VectorXd& signal, int rate, double releaseTime = 0.3) {
    double alpha = std::exp(-1.0 / (releaseTime * rate));
    Eigen::VectorXd vuLevel(signal.size());
    vuLevel[0] = std::abs(signal[0]);
    for (int i = 1; i < signal.size(); ++i) {
        vuLevel[i] = std::max(alpha * vuLevel[i - 1], std::abs(signal[i]));
    }
    return vuLevel;
}

// VU メーターのレベルを計算
Eigen::MatrixXd calculateVuLevels(const Eigen::MatrixXd& normalizedData, int rate) {
    int channels = normalizedData.cols();
    Eigen::MatrixXd vuLevels(normalizedData.rows(), channels);
    for (int ch = 0; ch < channels; ++ch) {
        vuLevels.col(ch) = vuMeter(normalizedData.col(ch), rate);
    }
    return vuLevels;
}

// 二分探索でゲインを調整
double adjustGain(const Eigen::MatrixXd& normalizedData, int rate, double refLevel = REF_LEVEL, double tol = 1e-4) {
    double low = 0.1, high = 10.0;
    while (high - low > tol) {
        double mid = (low + high) / 2.0;
        Eigen::MatrixXd vuLevels = calculateVuLevels(normalizedData * mid, rate);
        double maxVu = vuLevels.maxCoeff();
        if (maxVu < refLevel) {
            low = mid;
        } else {
            high = mid;
        }
    }
    return (low + high) / 2.0;
}

// オーディオデータを保存
void saveWavFile(const std::string& filePath, const Eigen::MatrixXd& data, int sampleRate) {
    int frames = data.rows();
    int channels = data.cols();
    std::vector<short> buffer(frames * channels);
    for (int i = 0; i < frames; ++i) {
        for (int j = 0; j < channels; ++j) {
            buffer[i * channels + j] = static_cast<short>(data(i, j) * std::numeric_limits<short>::max());
        }
    }
    SndfileHandle fileHandle(filePath, SFM_WRITE, SF_FORMAT_WAV | SF_FORMAT_PCM_16, channels, sampleRate);
    fileHandle.write(buffer.data(), frames * channels);
}

int main() {
    // ファイル選択（ここでは手動でファイルパスを入力）
    std::string filePath;
    std::cout << "Enter the path to the WAV file: ";
    std::cin >> filePath;

    // オーディオファイルの読み込み
    int sampleRate;
    Eigen::MatrixXd data = readWavFile(filePath, sampleRate);

    std::cout << "Normalized data range: " << data.minCoeff() << " to " << data.maxCoeff() << std::endl;

    // ゲインを調整
    double gainAdjustment = adjustGain(data, sampleRate);
    std::cout << "Gain adjustment factor: " << gainAdjustment << std::endl;

    // 統一したゲインを適用
    Eigen::MatrixXd adjustedData = data * gainAdjustment;
    std::cout << "Adjusted data range: " << adjustedData.minCoeff() << " to " << adjustedData.maxCoeff() << std::endl;

    // 新しいファイル名を生成
    std::string outputFilePath = filePath.substr(0, filePath.find_last_of('.')) + "_VUoutput.wav";

    // 調整後のオーディオデータを保存
    saveWavFile(outputFilePath, adjustedData, sampleRate);
    std::cout << "Adjusted audio saved as '" << outputFilePath << "'" << std::endl;

    return 0;
}
