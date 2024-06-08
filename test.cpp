struct SDL_Rect
{
    int x, y, w, h;
};
struct SDL_MouseButtonEvent
{
    int x, y;
};
struct SDL_MouseMotionEvent
{
    int x, y;
};
typedef void* SDL_Renderer;
inline void SDL_RenderFillRect(SDL_Renderer*, SDL_Rect*) {}
inline void SDL_RenderPresent(SDL_Renderer*) {}
inline void SDL_SetRenderDrawColor(SDL_Renderer*, int, int, int, int) {}
inline void SDL_RenderClear(SDL_Renderer*) {}
#include <pinktrombone.h>

using namespace pinktrombone;

void setRestDiameter(Tract& tract_)
{
    const double innerTongueControlRadius_ = 2.05;
    const double outerTongueControlRadius_ = 3.5;
    const double tongueLowerIndexBound_ = tract_.bladeStart() + 2;
    const double tongueUpperIndexBound_ = tract_.tipStart() - 3;
    double tongueIndex_ = (tongueLowerIndexBound_ + tongueUpperIndexBound_) / 2.;
    double tongueDiameter_ = (innerTongueControlRadius_ + outerTongueControlRadius_) / 2.;
    double gridOffset_ = 1.7;
    double noseOffset_ = 0.8;
    const double stepLength_ = 0.397;

    auto& restDiameter = tract_.restDiameter();
    for (int i = tract_.bladeStart(); i < tract_.lipStart(); i++)
    {
        double t = 1.1 * M_PI * (tongueIndex_ - i) / (tract_.tipStart() - tract_.bladeStart());
        double fixedTongueDiameter = 2 + (tongueDiameter_ - 2) / 1.5;
        double curve = (1.5 - fixedTongueDiameter + gridOffset_) * std::cos(t);
        if (i == tract_.bladeStart() - 2 || i == tract_.lipStart() - 1) curve *= 0.8;
        if (i == tract_.bladeStart() || i == tract_.lipStart() - 2) curve *= 0.94;
        restDiameter[i] = 1.5 - curve;
    }
}

int main(int argc, char *argv[])
{
    std::ifstream ifs("dump.bin", std::ios_base::binary);
    std::ofstream ofs("audio.bin", std::ios_base::binary);
    Glottis glottis_;
    Tract tract_{ glottis_, 44 };
    int blockLength_ = 512;
    int sampleRate = 44100;
    double blockTime_ = static_cast<double>(blockLength_) / sampleRate;

    setRestDiameter(tract_);
    tract_.targetDiameter() = tract_.restDiameter();
    std::vector<double> inputArray1(blockLength_);
    std::vector<double> inputArray2(blockLength_);
    std::vector<double> outArray(blockLength_);

    int i = 0;
    while (!ifs.eof())
    {
        std::vector<float> data(50);
        ifs.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof (float));
        // std::cout << data[0] << std::endl;
        std::vector<double> ddata(50);
        std::copy(data.begin(), data.end(), ddata.begin());
        // std::cout << ddata[2] << std::endl;
        glottis_.isTouched(ddata[1] > 0.0);
        glottis_.UIFrequency(ddata[2]);
        glottis_.UITenseness(ddata[3]);
        glottis_.loudness(data[4]);
        tract_.velumTarget(data[5]);
        auto& x = tract_.targetDiameter();
        std::copy(ddata.begin() + 6, ddata.end(), x.begin());
        // glottis_.loudness(std::pow(glottis_.UITenseness(), 0.25));

        glottis_.finishBlock();
        tract_.finishBlock(blockTime_);

        for (size_t j = 0; j < blockLength_; j++)
        {
            // TODO Apply 500Hz bandpass for aspirate
            inputArray1[j] = (static_cast<double>(std::rand()) / RAND_MAX) * 2. - 1.;
            // TODO Apply 1000Hz bandpass for fricative
            inputArray2[j] = (static_cast<double>(std::rand()) / RAND_MAX) * 2. - 1.;
        }

        for (size_t j = 0; j < blockLength_; j++)
        {
            double lambda1 = static_cast<double>(j) / blockLength_;
            double lambda2 = (j + 0.5) / blockLength_;
            double glottalOutput = glottis_.runStep(lambda1, inputArray1[j]);

            double vocalOutput = 0;
            // Tract runs at twice the sample rate 
            tract_.runStep(glottalOutput, inputArray2[j], lambda1);
            vocalOutput += tract_.lipOutput() + tract_.noseOutput();
            tract_.runStep(glottalOutput, inputArray2[j], lambda2);
            vocalOutput += tract_.lipOutput() + tract_.noseOutput();
            outArray[j] = vocalOutput * 0.125;
        }
        ofs.write(reinterpret_cast<char*>(outArray.data()), outArray.size() * sizeof (double));
    }
}
