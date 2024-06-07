/*

P I N K   T R O M B O N E

Bare-handed procedural speech synthesis

version 1.1, March 2017
by Neil Thapen
venuspatrol.nfshost.com


Bibliography

Julius O. Smith III, "Physical audio signal processing for virtual musical instruments and audio effects."
https://ccrma.stanford.edu/~jos/pasp/

Story, Brad H. "A parametric model of the vocal tract area function for vowel and consonant simulation."
The Journal of the Acoustical Society of America 117.5 (2005): 3231-3254.

Lu, Hui-Ling, and J. O. Smith. "Glottal source modeling for singing voice synthesis."
Proceedings of the 2000 International Computer Music Conference. 2000.

Mullen, Jack. Physical modelling of the vocal tract with the 2D digital waveguide mesh.
PhD thesis, University of York, 2006.


Copyright 2017 Neil Thapen

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.


Converted to C++ by Katsuya Iida.
Version 2024-06-01

*/

#include <algorithm>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>

namespace pinktrombone
{
    namespace noise {
        double simplex1(double x);
        double simplex2(double xin, double yin);
    }

    template<typename T>
    T clamp(T x, T a, T b)
    {
        return std::min(std::max(x, a), b);
    }

    double moveTowards(double current, double target, double amount)
    {
        if (current < target) return std::min(current + amount, target);
        else return std::max(current - amount, target);
    }

    double moveTowards(double current, double target, double amountUp, double amountDown)
    {
        if (current < target) return std::min(current + amountUp, target);
        else return std::max(current - amountDown, target);
    }


    bool autoWobble = false;
    bool alwaysVoice = false;
    int channels;
    const int sampleRate = 44100;
    double blockTime = 4096 / 44100.;

    struct Touch
    {
        std::string id;
        double startTime;
        double fricative_intensity;
        double endTime;
        bool alive;
        double index;
        double diameter;
    };

    class Glottis
    {
    private:
        double waveformLength_;
        double frequency_;
        double Rd_;
        double alpha_;
        double E0_;
        double epsilon_;
        double shift_;
        double Delta_;
        double Te_;
        double omega_;

        double timeInWaveform_ = 0;
        double oldFrequency_ = 140;
        double newFrequency_ = 140;
        double UIFrequency_ = 140;
        double smoothFrequency_ = 140;
        double oldTenseness_ = 0.6;
        double newTenseness_ = 0.6;
        const double UITenseness_;
        double totalTime_ = 0;
        double vibratoAmount_ = 0.005;
        double vibratoFrequency_ = 6;
        double intensity_ = 0;
        double loudness_ = 1;
        bool isTouched_ = false;
        std::string touch_;

    public:
        Glottis() : UITenseness_{ 1 - std::cos(0.6 * M_PI * 0.5) }
        {
            double baseNote = 87.3071; // F2
            double semitone = 8; // C#
            UIFrequency_ = baseNote * std::pow(2, semitone / 12);
            if (intensity_ == 0) smoothFrequency_ = UIFrequency_;
            //Glottis.UIRd = 3*local_y / (keyboardHeight-20);
            loudness_ = std::pow(UITenseness_, 0.25);
        }

        double runStep(double lambda, double noiseSource)
        {
            auto timeStep = 1.0 / sampleRate;
            timeInWaveform_ += timeStep;
            totalTime_ += timeStep;
            if (timeInWaveform_ > waveformLength_)
            {
                timeInWaveform_ -= waveformLength_;
                setupWaveform(lambda);
            }
            auto out = normalizedLFWaveform(timeInWaveform_ / waveformLength_);
            auto aspiration = intensity_ * (1 - std::sqrt(UITenseness_)) * getNoiseModulator() * noiseSource;
            aspiration *= 0.2 + 0.02 * noise::simplex1(totalTime_ * 1.99);
            out += aspiration;
            return out;
        }

        double getNoiseModulator()
        {
            auto voiced = 0.1 + 0.2 * std::max(0.0, std::sin(M_PI * 2 * timeInWaveform_ / waveformLength_));
            //return 0.3;
            return UITenseness_ * intensity_ * voiced + (1 - UITenseness_ * intensity_) * 0.3;
        }

        void finishBlock()
        {
            double vibrato = 0.;
            vibrato += vibratoAmount_ * std::sin(2 * M_PI * totalTime_ * vibratoFrequency_);
            vibrato += 0.02 * noise::simplex1(totalTime_ * 4.07);
            vibrato += 0.04 * noise::simplex1(totalTime_ * 2.15);
            if (autoWobble)
            {
                vibrato += 0.2 * noise::simplex1(totalTime_ * 0.98);
                vibrato += 0.4 * noise::simplex1(totalTime_ * 0.5);
            }
            if (UIFrequency_ > smoothFrequency_)
                smoothFrequency_ = std::min(smoothFrequency_ * 1.1, UIFrequency_);
            if (UIFrequency_ < smoothFrequency_)
                smoothFrequency_ = std::max(smoothFrequency_ / 1.1, UIFrequency_);
            oldFrequency_ = newFrequency_;
            newFrequency_ = smoothFrequency_ * (1 + vibrato);
            oldTenseness_ = newTenseness_;
            newTenseness_ = UITenseness_
                + 0.1 * noise::simplex1(totalTime_ * 0.46) + 0.05 * noise::simplex1(totalTime_ * 0.36);
            if (!isTouched_ && alwaysVoice) newTenseness_ += (3 - UITenseness_) * (1 - intensity_);

            if (isTouched_ || alwaysVoice) { intensity_ += 0.13; }
            else { intensity_ -= 0.05; }
            intensity_ = clamp(intensity_, 0., 1.);
        }

        void setupWaveform(double lambda)
        {
            frequency_ = oldFrequency_ * (1 - lambda) + newFrequency_ * lambda;
            auto tenseness = oldTenseness_ * (1 - lambda) + newTenseness_ * lambda;
            Rd_ = 3 * (1 - tenseness);
            waveformLength_ = 1.0 / frequency_;

            if (Rd_ < 0.5) Rd_ = 0.5;
            if (Rd_ > 2.7) Rd_ = 2.7;

            // normalized to time = 1, Ee = 1
            auto Ra = -0.01 + 0.048 * Rd_;
            auto Rk = 0.224 + 0.118 * Rd_;
            auto Rg = (Rk / 4) * (0.5 + 1.2 * Rk) / (0.11 * Rd_ - Ra * (0.5 + 1.2 * Rk));

            auto Ta = Ra;
            auto Tp = 1 / (2 * Rg);
            Te_ = Tp + Tp * Rk; //

            epsilon_ = 1 / Ta;
            shift_ = std::exp(-epsilon_ * (1 - Te_));
            Delta_ = 1 - shift_; //divide by this to scale RHS

            auto RHSIntegral = (1 / epsilon_) * (shift_ - 1) + (1 - Te_) * shift_;
            RHSIntegral = RHSIntegral / Delta_;

            auto totalLowerIntegral = -(Te_ - Tp) / 2 + RHSIntegral;
            auto totalUpperIntegral = -totalLowerIntegral;

            omega_ = M_PI / Tp;
            auto s = std::sin(omega_ * Te_);
            // need E0*e^(alpha*Te)*s = -1 (to meet the return at -1)
            // and E0*e^(alpha*Tp/2) * Tp*2/pi = totalUpperIntegral 
            //             (our approximation of the integral up to Tp)
            // writing x for e^alpha;
            // have E0*x^Te*s = -1 and E0 * x^(Tp/2) * Tp*2/pi = totalUpperIntegral
            // dividing the second by the first;
            // letting y = x^(Tp/2 - Te);
            // y * Tp*2 / (pi*s) = -totalUpperIntegral;
            auto y = -M_PI * s * totalUpperIntegral / (Tp * 2);
            auto z = std::log(y);
            alpha_ = z / (Tp / 2 - Te_);
            E0_ = -1 / (s * std::exp(alpha_ * Te_));
        }

        double normalizedLFWaveform(double t)
        {
            double output;
            if (t > Te_) output = (-std::exp(-epsilon_ * (t - Te_)) + shift_) / Delta_;
            else output = E0_ * std::exp(alpha_ * t) * std::sin(omega_ * t);

            return output * intensity_ * loudness_;
        }

        void handleTouches(const std::map<std::string, Touch>& touchesWithMouse)
        {
            if (!touch_.empty() && !touchesWithMouse.at(touch_).alive) touch_ = "";

            if (touch_.empty())
            {
                for (auto it = touchesWithMouse.begin(); it != touchesWithMouse.end(); ++it)
                {
                    auto& touch = it->second;
                    if (!touch.alive) continue;
                    // TODO if (touch.y < this.keyboardTop) continue;
                    touch_ = it->second.id;
                }
            }

            if (!touch_.empty())
            {
                // TODO
#if 0
                var local_y = this.touch.y - this.keyboardTop - 10;
                var local_x = this.touch.x - this.keyboardLeft;
                local_y = Math.clamp(local_y, 0, this.keyboardHeight - 26);
                var semitone = this.semitones * local_x / this.keyboardWidth + 0.5;
                Glottis.UIFrequency = this.baseNote * Math.pow(2, semitone / 12);
                if (Glottis.intensity == 0) Glottis.smoothFrequency = Glottis.UIFrequency;
                //Glottis.UIRd = 3*local_y / (this.keyboardHeight-20);
                var t = Math.clamp(1 - local_y / (this.keyboardHeight - 28), 0, 1);
                Glottis.UITenseness = 1 - Math.cos(t * Math.PI * 0.5);
                Glottis.loudness = Math.pow(Glottis.UITenseness, 0.25);
                this.x = this.touch.x;
                this.y = local_y + this.keyboardTop + 10;
#endif
            }

            isTouched_ = !touch_.empty();
        }
    };

    class Tract
    {
        struct Transient
        {
            int position;
            double timeAlive;
            double lifeTime;
            double strength;
            double exponent;
        };

        Glottis& glottis;
        int n_ = 44;
        int bladeStart_ = 10;
        int tipStart_ = 32;
        int lipStart_ = 39;
        std::vector<double> R_; //component going right
        std::vector<double> L_; //component going left
        std::vector<double> reflection_;
        std::vector<double> newReflection_;
        std::vector<double> junctionOutputR_;
        std::vector<double> junctionOutputL_;
        std::vector<double> maxAmplitude_;
        std::vector<double> diameter_;
        std::vector<double> restDiameter_;
        std::vector<double> targetDiameter_;
        std::vector<double> newDiameter_;
        std::vector<double> A_;
        double glottalReflection_ = 0.75;
        double lipReflection_ = -0.85;
        int lastObstruction_ = -1;
        double fade_ = 1.0; //0.9999;
        double movementSpeed_ = 15; //cm per second
        std::vector<Transient> transients_;
        double lipOutput_ = 0.;
        double noseOutput_ = 0.;
        double velumTarget_ = 0.01;

        int noseLength_;
        int noseStart_;
        std::vector<double> noseR_;
        std::vector<double> noseL_;
        std::vector<double> noseJunctionOutputR_;
        std::vector<double> noseJunctionOutputL_;
        std::vector<double> noseReflection_;
        std::vector<double> noseDiameter_;
        std::vector<double> noseA_;
        std::vector<double> noseMaxAmplitude_;

        double newReflectionLeft_;
        double newReflectionRight_;
        double newReflectionNose_;
        double reflectionLeft_;
        double reflectionRight_;
        double reflectionNose_;

    public:
        Tract(Glottis& glottis) : glottis(glottis)
        {
            this->bladeStart_ = this->bladeStart_ * n_ / 44;
            this->tipStart_ = this->tipStart_ * n_ / 44;
            this->lipStart_ = this->lipStart_ * n_ / 44;
            this->diameter_.resize(n_);
            this->restDiameter_.resize(n_);
            this->targetDiameter_.resize(n_);
            newDiameter_.resize(n_);
            for (auto i = 0; i < n_; i++)
            {
                double diameter = 0.;
                if (i < 7 * n_ / 44 - 0.5) diameter = 0.6;
                else if (i < 12 * n_ / 44) diameter = 1.1;
                else diameter = 1.5;
                this->diameter_[i] = this->restDiameter_[i] = this->targetDiameter_[i] = newDiameter_[i] = diameter;
            }
            this->R_.resize(n_);
            this->L_.resize(n_);
            this->reflection_.resize(n_ + 1);
            newReflection_.resize(n_ + 1);
            this->junctionOutputR_.resize(n_ + 1);
            this->junctionOutputL_.resize(n_ + 1);
            this->A_.resize(n_);
            this->maxAmplitude_.resize(n_);

            noseLength_ = 28 * n_ / 44;
            noseStart_ = n_ - noseLength_ + 1;
            noseR_.resize(noseLength_);
            noseL_.resize(noseLength_);
            noseJunctionOutputR_.resize(noseLength_ + 1);
            noseJunctionOutputL_.resize(noseLength_ + 1);
            noseReflection_.resize(noseLength_ + 1);
            noseDiameter_.resize(noseLength_);
            noseA_.resize(noseLength_);
            noseMaxAmplitude_.resize(noseLength_);
            for (auto i = 0; i < noseLength_; i++)
            {
                double diameter;
                double d = 2. * i / noseLength_;
                if (d < 1) diameter = 0.4 + 1.6 * d;
                else diameter = 0.5 + 1.5 * (2 - d);
                diameter = std::min(diameter, 1.9);
                // diameter = 2.3 * std::sin(M_PI * i / _noseLength) + .2;
                noseDiameter_[i] = diameter;
            }
            newReflectionLeft_ = newReflectionRight_ = newReflectionNose_ = 0;
            this->calculateReflections();
            this->calculateNoseReflections();
            noseDiameter_[0] = this->velumTarget_;
        }

        int n() const { return n_; }
        int bladeStart() const { return bladeStart_; }
        int lipStart() const { return lipStart_; }
        int tipStart() const { return tipStart_; }
        double lipOutput() const { return lipOutput_; }
        double noseLength() const { return noseLength_; }
        int noseStart() const { return noseStart_; }
        double noseOutput() const { return noseOutput_; }
        const std::vector<double>& diameter() const { return diameter_; }
        std::vector<double>& restDiameter() { return restDiameter_; }
        std::vector<double>& targetDiameter() { return targetDiameter_; }
        const std::vector<double>& noseDiameter() const { return noseDiameter_; }
        void velumTarget(double value) { velumTarget_ = value; }

        void reshapeTract(double deltaTime)
        {
            double amount = deltaTime * this->movementSpeed_;
            int newLastObstruction = -1;
            for (auto i = 0; i < n_; i++)
            {
                auto diameter = this->diameter_[i];
                auto targetDiameter = this->targetDiameter_[i];
                if (diameter <= 0) newLastObstruction = i;
                double slowReturn;
                if (i < noseStart_) slowReturn = 0.6;
                else if (i >= this->tipStart_) slowReturn = 1.0;
                else slowReturn = 0.6 + 0.4 * (i - noseStart_) / (this->tipStart_ - noseStart_);
                this->diameter_[i] = moveTowards(diameter, targetDiameter, slowReturn * amount, 2 * amount);
            }
            if (this->lastObstruction_ > -1 && newLastObstruction == -1 && noseA_[0] < 0.05)
            {
                this->addTransient(this->lastObstruction_);
            }
            this->lastObstruction_ = newLastObstruction;

            amount = deltaTime * this->movementSpeed_;
            noseDiameter_[0] = moveTowards(noseDiameter_[0], this->velumTarget_,
                amount * 0.25, amount * 0.1);
            noseA_[0] = noseDiameter_[0] * noseDiameter_[0];
        }

        void calculateReflections()
        {
            for (auto i = 0; i < n_; i++)
            {
                this->A_[i] = this->diameter_[i] * this->diameter_[i]; //ignoring PI etc.
            }
            for (auto i = 1; i < n_; i++)
            {
                this->reflection_[i] = newReflection_[i];
                if (this->A_[i] == 0) newReflection_[i] = 0.999; //to prevent some bad behaviour if 0
                else newReflection_[i] = (this->A_[i - 1] - this->A_[i]) / (this->A_[i - 1] + this->A_[i]);
            }

            //now at junction with nose

            this->reflectionLeft_ = newReflectionLeft_;
            this->reflectionRight_ = newReflectionRight_;
            this->reflectionNose_ = newReflectionNose_;
            auto sum = this->A_[noseStart_] + this->A_[noseStart_ + 1] + noseA_[0];
            newReflectionLeft_ = (2 * this->A_[noseStart_] - sum) / sum;
            newReflectionRight_ = (2 * this->A_[noseStart_ + 1] - sum) / sum;
            newReflectionNose_ = (2 * noseA_[0] - sum) / sum;
        }

        void calculateNoseReflections()
        {
            for (auto i = 0; i < noseLength_; i++)
            {
                noseA_[i] = noseDiameter_[i] * noseDiameter_[i];
            }
            for (auto i = 1; i < noseLength_; i++)
            {
                noseReflection_[i] = (noseA_[i - 1] - noseA_[i]) / (noseA_[i - 1] + noseA_[i]);
            }
        }

        void runStep(double glottalOutput, double turbulenceNoise, double lambda)
        {
            auto updateAmplitudes = (static_cast<double>(std::rand()) / RAND_MAX < 0.1);

            //mouth
            this->processTransients();
            this->addTurbulenceNoise(turbulenceNoise);

            //this->_glottalReflection = -0.8 + 1.6 * Glottis.newTenseness;
            this->junctionOutputR_[0] = this->L_[0] * this->glottalReflection_ + glottalOutput;
            this->junctionOutputL_[n_] = this->R_[n_ - 1] * this->lipReflection_;

            for (auto i = 1; i < n_; i++)
            {
                auto r = this->reflection_[i] * (1 - lambda) + newReflection_[i] * lambda;
                auto w = r * (this->R_[i - 1] + this->L_[i]);
                this->junctionOutputR_[i] = this->R_[i - 1] - w;
                this->junctionOutputL_[i] = this->L_[i] + w;
            }

            //now at junction with nose
            auto i = noseStart_;
            auto r = newReflectionLeft_ * (1 - lambda) + this->reflectionLeft_ * lambda;
            this->junctionOutputL_[i] = r * this->R_[i - 1] + (1 + r) * (noseL_[0] + this->L_[i]);
            r = newReflectionRight_ * (1 - lambda) + this->reflectionRight_ * lambda;
            this->junctionOutputR_[i] = r * this->L_[i] + (1 + r) * (this->R_[i - 1] + noseL_[0]);
            r = newReflectionNose_ * (1 - lambda) + this->reflectionNose_ * lambda;
            noseJunctionOutputR_[0] = r * noseL_[0] + (1 + r) * (this->L_[i] + this->R_[i - 1]);

            for (auto i = 0; i < n_; i++)
            {
                this->R_[i] = this->junctionOutputR_[i] * 0.999;
                this->L_[i] = this->junctionOutputL_[i + 1] * 0.999;

                //this->_R[i] = std::clamp(this->_junctionOutputR[i] * this->_fade, -1, 1);
                //this->_L[i] = std::clamp(this->_junctionOutputL[i+1] * this->_fade, -1, 1);    

                if (updateAmplitudes)
                {
                    auto amplitude = std::abs(this->R_[i] + this->L_[i]);
                    if (amplitude > this->maxAmplitude_[i]) this->maxAmplitude_[i] = amplitude;
                    else this->maxAmplitude_[i] *= 0.999;
                }
            }

            this->lipOutput_ = this->R_[n_ - 1];

            //nose     
            noseJunctionOutputL_[noseLength_] = noseR_[noseLength_ - 1] * this->lipReflection_;

            for (auto i = 1; i < noseLength_; i++)
            {
                auto w = noseReflection_[i] * (noseR_[i - 1] + noseL_[i]);
                noseJunctionOutputR_[i] = noseR_[i - 1] - w;
                noseJunctionOutputL_[i] = noseL_[i] + w;
            }

            for (auto i = 0; i < noseLength_; i++)
            {
                noseR_[i] = noseJunctionOutputR_[i] * this->fade_;
                noseL_[i] = noseJunctionOutputL_[i + 1] * this->fade_;

                //_noseR[i] = std::clamp(_noseJunctionOutputR[i] * this->_fade, -1, 1);
                //_noseL[i] = std::clamp(_noseJunctionOutputL[i+1] * this->_fade, -1, 1);    

                if (updateAmplitudes)
                {
                    auto amplitude = std::abs(noseR_[i] + noseL_[i]);
                    if (amplitude > noseMaxAmplitude_[i]) noseMaxAmplitude_[i] = amplitude;
                    else noseMaxAmplitude_[i] *= 0.999;
                }
            }

            noseOutput_ = noseR_[noseLength_ - 1];

        }

        void finishBlock()
        {
            this->reshapeTract(blockTime);
            this->calculateReflections();
        }

        void addTransient(int position)
        {
            Transient trans;
            trans.position = position;
            trans.timeAlive = 0;
            trans.lifeTime = 0.2;
            trans.strength = 0.3;
            trans.exponent = 200;
            this->transients_.push_back(trans);
        }

        void processTransients()
        {
            for (auto i = 0; i < this->transients_.size(); i++)
            {
                auto trans = this->transients_[i];
                auto amplitude = trans.strength * std::pow(2, -trans.exponent * trans.timeAlive);
                this->R_[trans.position] += amplitude / 2;
                this->L_[trans.position] += amplitude / 2;
                trans.timeAlive += 1.0 / (sampleRate * 2);
            }
            for (auto it = transients_.begin(); it != transients_.end();)
            {
                if (it->timeAlive > it->lifeTime)
                {
                    it = transients_.erase(it);
                }
                else
                {
                    ++it;
                }
            }
        }

        double turbulenceIntensity_ = 0.;
        double turbulenceIndex_ = 0.;
        double turbulenceDiameter_ = 0.;

        void turbulanceNoise(double intensity, double index, double diameter)
        {
            turbulenceIntensity_ = intensity;
            turbulenceIndex_ = index;
            turbulenceDiameter_ = diameter;
        }

        void addTurbulenceNoise(double turbulenceNoise)
        {
            if (turbulenceIntensity_ == 0) return;
            this->addTurbulenceNoiseAtIndex(0.66 * turbulenceNoise * turbulenceIntensity_, turbulenceIndex_, turbulenceDiameter_);
        }

        void addTurbulenceNoiseAtIndex(double turbulenceNoise, double index, double diameter)
        {
            int i = static_cast<int>(index);
            double delta = index - i;
            turbulenceNoise *= glottis.getNoiseModulator();
            double thinness0 = clamp(8 * (0.7 - diameter), 0., 1.);
            double openness = clamp(30 * (diameter - 0.3), 0., 1.);
            double noise0 = turbulenceNoise * (1 - delta) * thinness0 * openness;
            double noise1 = turbulenceNoise * delta * thinness0 * openness;
            this->R_[i + 1] += noise0 / 2;
            this->L_[i + 1] += noise0 / 2;
            this->R_[i + 2] += noise1 / 2;
            this->L_[i + 2] += noise1 / 2;
        }
    };

    class TractUI {
    public:
        TractUI(Tract& tract) : tract{ tract }
        {

        }
        double getIndex(int x, int y)
        {
            return static_cast<double>(tract.n() * (x - tractRect.x)) / tractRect.w;
        }

        double getDiameter(int x, int y)
        {
            double v = static_cast<double>(y - (tractRect.y + (tractRect.h - 10) / 2 + 10));
            return static_cast<double>(v) * (tract.n() * stepLength) / tractRect.w;
        }

        void setRestDiameter()
        {
            auto& restDiameter = tract.restDiameter();
            for (int i = tract.bladeStart(); i < tract.lipStart(); i++)
            {
                double t = 1.1 * M_PI * (tongueIndex - i) / (tract.tipStart() - tract.bladeStart());
                double fixedTongueDiameter = 2 + (tongueDiameter - 2) / 1.5;
                double curve = (1.5 - fixedTongueDiameter + gridOffset) * std::cos(t);
                if (i == tract.bladeStart() - 2 || i == tract.lipStart() - 1) curve *= 0.8;
                if (i == tract.bladeStart() || i == tract.lipStart() - 2) curve *= 0.94;
                restDiameter[i] = 1.5 - curve;
            }
        }

        void handleTouches(std::map<std::string, Touch>& touchesWithMouse)
        {

            double tongueIndexCentre = 0.5 * (tongueLowerIndexBound + tongueUpperIndexBound);

            if (!tongueTouch.empty() && !touchesWithMouse[tongueTouch].alive) tongueTouch = "";

            if (tongueTouch.empty()) {
                for (auto it = touchesWithMouse.begin(); it != touchesWithMouse.end(); ++it)
                {
                    auto& touch = it->second;
                    if (!touch.alive) continue;
                    if (touch.fricative_intensity == 1.) continue; //only new touches will pass this
                    double index = touch.index;
                    double diameter = touch.diameter;
                    if (index >= tongueLowerIndexBound - 4 && index <= tongueUpperIndexBound + 4
                        && diameter >= innerTongueControlRadius - 0.5 && diameter <= outerTongueControlRadius + 0.5)
                    {
                        tongueTouch = touch.id;
                    }
                }
            }

            if (!tongueTouch.empty())
            {
                auto& touch = touchesWithMouse.at(tongueTouch);
                double index = touch.index;
                double diameter = touch.diameter;
                double fromPoint = (outerTongueControlRadius - diameter) / (outerTongueControlRadius - innerTongueControlRadius);
                fromPoint = clamp(fromPoint, 0., 1.);
                fromPoint = std::pow(fromPoint, 0.58) - 0.2 * (fromPoint * fromPoint - fromPoint); //horrible kludge to fit curve to straight line
                tongueDiameter = clamp(diameter, innerTongueControlRadius, outerTongueControlRadius);
                //tongueIndex = Math.clamp(index, tongueLowerIndexBound, tongueUpperIndexBound);
                double out = fromPoint * 0.5 * (tongueUpperIndexBound - tongueLowerIndexBound);
                tongueIndex = clamp(index, tongueIndexCentre - out, tongueIndexCentre + out);
            }

            setRestDiameter();
            tract.targetDiameter() = tract.restDiameter();

            //other constrictions and nose
            tract.velumTarget(0.01);
            for (auto it = touchesWithMouse.begin(); it != touchesWithMouse.end(); ++it)
            {
                auto& touch = it->second;
                if (!touch.alive) continue;
                double index = touch.index;
                double diameter = touch.diameter;
                if (index > tract.noseStart() && diameter < -noseOffset)
                {
                    tract.velumTarget(0.4);
                }
                if (diameter < -0.85 - noseOffset) continue;
                diameter -= 0.3;
                if (diameter < 0) diameter = 0;
                double width = 2.;
                if (index < 25) width = 10;
                else if (index >= tract.tipStart()) width = 5;
                else width = 10 - 5 * (index - 25) / (tract.tipStart() - 25);
                if (index >= 2 && index < tract.n() && diameter < 3)
                {
                    int intIndex = static_cast<int>(std::round(index));
                    for (int i = -static_cast<int>(std::ceil(width)) - 1; i < width + 1; i++)
                    {
                        if (intIndex + i < 0 || intIndex + i >= tract.n()) continue;
                        double relpos = (intIndex + i) - index;
                        relpos = std::abs(relpos) - 0.5;
                        double shrink;
                        if (relpos <= 0) shrink = 0;
                        else if (relpos > width) shrink = 1;
                        else shrink = 0.5 * (1 - std::cos(M_PI * relpos / width));
                        if (diameter < tract.targetDiameter()[intIndex + i])
                        {
                            tract.targetDiameter()[intIndex + i] = diameter + (tract.targetDiameter()[intIndex + i] - diameter) * shrink;
                        }
                    }
                }
            }

            // addTurbulenceNoise()
            for (auto it = touchesWithMouse.begin(); it != touchesWithMouse.end(); ++it)
            {
                auto& touch = it->second;
                if (touch.index<2 || touch.index>tract.n()) continue;
                if (touch.diameter <= 0) continue;
                tract.turbulanceNoise(touch.fricative_intensity, touch.index, touch.diameter);
            }
        }

        void draw(SDL_Renderer* renderer)
        {
            SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
            SDL_RenderClear(renderer);
            SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
            auto& diameter = tract.diameter();
            for (size_t i = 0; i < diameter.size(); i++) {
                if (i == tract.noseStart()) {
                    SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
                }
                else {
                    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
                }
                const double v = diameter[i] * tractRect.w / (diameter.size() * stepLength);
                SDL_Rect rect{
                    static_cast<int>(tractRect.x + tractRect.w * i / diameter.size()),
                    tractRect.y + (tractRect.h - 10) / 2 + 10,
                    static_cast<int>(tractRect.w * (i + 1) / diameter.size() - tractRect.w * i / diameter.size()),
                    static_cast<int>(v)
                };
                SDL_RenderFillRect(renderer, &rect);
            }

            auto noseDiameter = tract.noseDiameter();
            for (int i = 0; i < noseDiameter.size(); i++) {
                SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
                const double v = noseDiameter[i] * tractRect.w / (diameter.size() * stepLength);
                SDL_Rect rect{
                    static_cast<int>(tractRect.x + tractRect.w * (tract.noseStart() + i) / diameter.size()),
                    tractRect.y + (tractRect.h - 10) / 2 - static_cast<int>(v),
                    static_cast<int>(tractRect.w * (tract.noseStart() + i + 1) / diameter.size() - tractRect.w * (tract.noseStart() + i) / diameter.size()),
                    static_cast<int>(v)
                };
                SDL_RenderFillRect(renderer, &rect);
            }

            SDL_RenderPresent(renderer);
        }

    private:
        Tract& tract;
        const double innerTongueControlRadius = 2.05;
        const double outerTongueControlRadius = 3.5;
        const double tongueLowerIndexBound = tract.bladeStart() + 2;
        const double tongueUpperIndexBound = tract.tipStart() - 3;
        SDL_Rect tractRect{ 20, 20, 600, 200 };
        std::string tongueTouch;
        double tongueIndex = (tongueLowerIndexBound + tongueUpperIndexBound) / 2.;
        double tongueDiameter = (innerTongueControlRadius + outerTongueControlRadius) / 2.;
        double gridOffset = 1.7;
        double noseOffset = 0.8;
        const double stepLength = 0.397;
    };

    class UI
    {
    public:
        UI() : glottis{}, tract{ glottis }, tractUI{ tract }
        {
        }

        void startMouse(const SDL_MouseButtonEvent* event)
        {
#if 0
            if (!AudioSystem.started)
            {
                AudioSystem.started = true;
                AudioSystem.startSound();
            }
#endif
            Touch touch = {};
            touch.startTime = std::clock() / static_cast<double>(CLOCKS_PER_SEC);
            touch.fricative_intensity = 0.;
            touch.endTime = 0.;
            touch.alive = true;
            touch.id = "mouse" + std::to_string(std::rand());
            touch.index = tractUI.getIndex(event->x, event->y);
            touch.diameter = tractUI.getDiameter(event->x, event->y);
            mouseTouch = touch.id;
            touchesWithMouse[touch.id] = touch;
            handleTouches();
        }

        void moveMouse(const SDL_MouseMotionEvent* event)
        {
            if (mouseTouch.empty()) return;
            auto& touch = touchesWithMouse.at(mouseTouch);
            if (!touch.alive) return;
            touch.index = tractUI.getIndex(event->x, event->y);
            touch.diameter = tractUI.getDiameter(event->x, event->y);
            handleTouches();
        }

        void endMouse(const SDL_MouseButtonEvent* event)
        {
            if (mouseTouch.empty()) return;
            double time = std::clock() / static_cast<double>(CLOCKS_PER_SEC);
            auto& touch = touchesWithMouse.at(mouseTouch);
            mouseTouch = "";
            if (!touch.alive) return;
            touch.alive = false;
            touch.endTime = time;
            handleTouches();
        }

        void handleTouches()
        {
            tractUI.handleTouches(touchesWithMouse);
            glottis.handleTouches(touchesWithMouse);
        }

        void updateTouches()
        {
            double time = std::clock() / static_cast<double>(CLOCKS_PER_SEC);
            double fricativeAttackTime = 0.1;
            for (auto it = touchesWithMouse.begin(); it != touchesWithMouse.end();)
            {
                auto& touch = it->second;
                if (!(touch.alive) && (time > touch.endTime + 1))
                {
                    it = touchesWithMouse.erase(it);
                }
                else if (touch.alive)
                {
                    ++it;
                    std::cout << touch.fricative_intensity << std::endl;
                    touch.fricative_intensity = clamp((time - touch.startTime) / fricativeAttackTime, 0., 1.);
                }
                else
                {
                    ++it;
                    std::cout << touch.fricative_intensity << std::endl;
                    touch.fricative_intensity = clamp(1 - (time - touch.endTime) / fricativeAttackTime, 0., 1.);
                }
            }
        }

        Glottis glottis;
        Tract tract;
        TractUI tractUI;
        std::map<std::string, Touch> touchesWithMouse;
        std::string mouseTouch;
    };
}
