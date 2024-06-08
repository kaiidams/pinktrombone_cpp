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

int main(int argc, char *argv[])
{
    std::ofstream ofs("audio.bin", std::ios_base::binary);
    Glottis glottis_;
    Tract tract_{ glottis_, 44 };
    int blockLength_ = 512;
    int sampleRate = 44100;
    double blockTime_ = static_cast<double>(blockLength_) / sampleRate;
    std::vector<double> inputArray1(blockLength_);
    std::vector<double> inputArray2(blockLength_);
    std::vector<double> outArray(blockLength_);
    for (size_t j = 0; j < blockLength_; j++)
    {
        // TODO Apply 500Hz bandpass for aspirate
        inputArray1[j] = (static_cast<double>(std::rand()) / RAND_MAX) * 2. - 1.;
        // TODO Apply 1000Hz bandpass for fricative
        inputArray2[j] = (static_cast<double>(std::rand()) / RAND_MAX) * 2. - 1.;
    }
    int i = 0;
    while (true)
    {
        if (i++ >= 1000)
        {
            break;
        }
        glottis_.isTouched(true);
        glottis_.UIFrequency(160.);
        glottis_.UITenseness(0.8);
        glottis_.UIFrequency(0.8);

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
        ofs.write(reinterpret_cast<char*>(outArray.data()), outArray.size() * sizeof (float));
        glottis_.finishBlock();
        tract_.finishBlock(blockTime_);
    }
}
