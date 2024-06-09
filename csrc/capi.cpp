#include <pinktrombone.h>
#include "capi.h"

struct PinkTrombone {
    PinkTrombone(int n) : glottis_{}, tract_{ glottis_, n }
    {
        blockTime_ = static_cast<double>(blockLength_) / sampleRate;
        inputArray1_.resize(blockLength_);
        inputArray2_.resize(blockLength_);
    }
    int control(double* data, size_t len);
    int process(double* data, size_t len);
    pinktrombone::Glottis glottis_;
    pinktrombone::Tract tract_;
    std::vector<double> inputArray1_;
    std::vector<double> inputArray2_;
    int sampleRate = 44100;
    int blockLength_ = 512;
    double blockTime_;
};

PinkTrombone* PinkTrombone_new(int n)
{
    return new PinkTrombone(n);
}

void PinkTrombone_delete(PinkTrombone* this_)
{
    delete this_;
}

int PinkTrombone::control(double* data, size_t len)
{
    auto& targetDiameter = tract_.targetDiameter();
    if (len != targetDiameter.size() + 5) return -1;
    glottis_.isTouched(data[0] > 0.0);
    glottis_.UIFrequency(data[1]);
    glottis_.UITenseness(data[2]);
    glottis_.loudness(data[3]);
    // glottis_.loudness(std::pow(glottis_.UITenseness(), 0.25));
    tract_.velumTarget(data[4]);
    std::copy(data + 5, data + 5 + targetDiameter.size(), targetDiameter.begin());
    return 0;
}

int PinkTrombone::process(double* data, size_t len)
{
    if (len != blockLength_) {
        return -1;
    }

    for (size_t j = 0; j < blockLength_; j++) {
        // TODO Apply 500Hz bandpass for aspirate
        inputArray1_[j] = (static_cast<double>(std::rand()) / RAND_MAX) * 2. - 1.;
        // TODO Apply 1000Hz bandpass for fricative
        inputArray2_[j] = (static_cast<double>(std::rand()) / RAND_MAX) * 2. - 1.;
    }

    for (size_t j = 0; j < blockLength_; j++) {
        double lambda1 = static_cast<double>(j) / blockLength_;
        double lambda2 = (j + 0.5) / blockLength_;
        double glottalOutput = glottis_.runStep(lambda1, inputArray1_[j]);

        double vocalOutput = 0;
        // Tract runs at twice the sample rate 
        tract_.runStep(glottalOutput, inputArray2_[j], lambda1);
        vocalOutput += tract_.lipOutput() + tract_.noseOutput();
        tract_.runStep(glottalOutput, inputArray2_[j], lambda2);
        vocalOutput += tract_.lipOutput() + tract_.noseOutput();
        data[j] = vocalOutput * 0.125;
    }

    glottis_.finishBlock();
    tract_.finishBlock(blockTime_);
    return 0;
}

int PinkTrombone_control(PinkTrombone* this_, double* data, size_t len)
{
    return this_->control(data, len);
}

int PinkTrombone_process(PinkTrombone* this_, double* data, size_t len)
{
    return this_->process(data, len);
}
