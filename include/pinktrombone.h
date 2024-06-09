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
#include <map>
#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>
#ifdef USE_SDL
#include <SDL.h>
#endif

namespace pinktrombone
{
    namespace noise
    {
        double simplex1(double x);
        double simplex2(double xin, double yin);
    }

    namespace math
    {
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
    }

    bool autoWobble = true;
    bool alwaysVoice = false;
    const int sampleRate = 44100;

    struct Touch
    {
        std::string id;
        double startTime;
        double fricative_intensity;
        double endTime;
        bool alive;
        int x;
        int y;
        double index;
        double diameter;
    };

    class Glottis
    {
    public:
        Glottis() : UITenseness_{ 1 - std::cos(0.6 * M_PI * 0.5) }
        {
            double semitone = 8.; // C#
            UIFrequency_ = baseNote_ * std::pow(2., semitone / 12.);
            if (intensity_ == 0) smoothFrequency_ = UIFrequency_;
            //Glottis.UIRd = 3*local_y / (keyboardHeight-20);
            loudness_ = std::pow(UITenseness_, 0.25);
        }

        void isTouched(bool value) { isTouched_ = value; }
        bool isTouched() const { return isTouched_; }
        void UIFrequency(double value) { UIFrequency_ = value; }
        double UIFrequency() const { return UIFrequency_; }
        void UITenseness(double value) { UITenseness_ = value; }
        double UITenseness() const { return UITenseness_; }
        void loudness(double value) { loudness_ = value; }
        double loudness() const { return loudness_; }

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

        double getNoiseModulator() const
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
            intensity_ = math::clamp(intensity_, 0., 1.);
        }

    private:
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

    public:
        void handleTouches(const std::map<std::string, Touch>& touchesWithMouse)
        {
#ifdef USE_SDL
            if (!touch_.empty() && !touchesWithMouse.at(touch_).alive) touch_ = "";

            if (touch_.empty())
            {
                for (auto it = touchesWithMouse.begin(); it != touchesWithMouse.end(); ++it)
                {
                    auto& touch = it->second;
                    if (!touch.alive) continue;
                    if (touch.y < keyboardRect_.y) continue;
                    touch_ = it->second.id;
                }
            }

            if (!touch_.empty())
            {
                auto& touch = touchesWithMouse.at(touch_);
                int local_y = touch.y - keyboardRect_.y - 10;
                int local_x = touch.x - keyboardRect_.x;
                local_y = math::clamp(local_y, 0, keyboardRect_.h - 26);
                double semitone = semitones_ * local_x / keyboardRect_.w + 0.5;
                UIFrequency_ = baseNote_ * std::pow(2, semitone / 12);
                if (intensity_ == 0) smoothFrequency_ = UIFrequency_;
                //Glottis.UIRd = 3*local_y / (this.keyboardHeight-20);
                double t = math::clamp(1. - static_cast<double>(local_y) / (keyboardRect_.h - 28), 0., 1.);
                UITenseness_ = 1. - std::cos(t * M_PI * 0.5);
                loudness_ = std::pow(UITenseness_, 0.25);
                x_ = touch.x;
                y_ = local_y + keyboardRect_.y + 10;
            }

            isTouched_ = !touch_.empty();
            // TODO
            for (auto it = touchesWithMouse.begin(); it != touchesWithMouse.end(); ++it)
            {
                auto& touch = it->second;
                if (touch.alive) {
                    isTouched_ = true;
                    break;
                }
            }
#endif
        }

    private:
        double waveformLength_ = 0.;
        double frequency_ = 0.;
        double Rd_ = 0.;
        double alpha_ = 0.;
        double E0_ = 0.;
        double epsilon_ = 0.;
        double shift_ = 0.;
        double Delta_ = 0.;
        double Te_ = 0.;
        double omega_ = 0.;

        double timeInWaveform_ = 0;
        double oldFrequency_ = 140;
        double newFrequency_ = 140;
        double smoothFrequency_ = 140;
        double oldTenseness_ = 0.6;
        double newTenseness_ = 0.6;
        double totalTime_ = 0;
        double vibratoAmount_ = 0.005;
        double vibratoFrequency_ = 6;
        double intensity_ = 0;
        double loudness_ = 1;

        // Control
        double UIFrequency_ = 140;
        double UITenseness_;
        bool isTouched_ = false;

        // UI
        std::string touch_;
        int x_ = 0;
        int y_ = 0;
        const int semitones_ = 20;
        const double baseNote_ = 87.3071; //F
#ifdef USE_SDL
        const SDL_Rect keyboardRect_{ 20, 240, 600, 100 };
#endif
    };

    class Tract
    {
    private:
        struct Transient
        {
            int position;
            double timeAlive;
            double lifeTime;
            double strength;
            double exponent;
        };

    public:
        Tract(const Glottis& glottis, int n) : glottis_{ glottis }, n_{ n }
        {
            bladeStart_ = bladeStart_ * n_ / 44;
            tipStart_ = tipStart_ * n_ / 44;
            lipStart_ = lipStart_ * n_ / 44;
            diameter_.resize(n_);
            restDiameter_.resize(n_);
            targetDiameter_.resize(n_);
            newDiameter_.resize(n_);
            for (auto i = 0; i < n_; i++)
            {
                double diameter = 0.;
                if (i < 7. * n_ / 44. - 0.5) diameter = 0.6;
                else if (i < 12. * n_ / 44.) diameter = 1.1;
                else diameter = 1.5;
                diameter_[i] = restDiameter_[i] = targetDiameter_[i] = newDiameter_[i] = diameter;
            }
            R_.resize(n_);
            L_.resize(n_);
            reflection_.resize(n_ + 1);
            newReflection_.resize(n_ + 1);
            junctionOutputR_.resize(n_ + 1);
            junctionOutputL_.resize(n_ + 1);
            A_.resize(n_);
            maxAmplitude_.resize(n_);

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
                noseDiameter_[i] = diameter;
            }
            newReflectionLeft_ = newReflectionRight_ = newReflectionNose_ = 0;
            calculateReflections();
            calculateNoseReflections();
            noseDiameter_[0] = velumTarget_;
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
        double velumTarget() const { return velumTarget_; }

        void turbulanceNoise(double intensity, double index, double diameter)
        {
            turbulenceIntensity_ = intensity;
            turbulenceIndex_ = index;
            turbulenceDiameter_ = diameter;
        }

    private:
        void reshapeTract(double deltaTime)
        {
            double amount = deltaTime * movementSpeed_;
            int newLastObstruction = -1;
            for (auto i = 0; i < n_; i++)
            {
                double diameter = diameter_[i];
                double targetDiameter = targetDiameter_[i];
                if (diameter <= 0) newLastObstruction = i;
                double slowReturn;
                if (i < noseStart_) slowReturn = 0.6;
                else if (i >= tipStart_) slowReturn = 1.0;
                else slowReturn = 0.6 + 0.4 * (i - noseStart_) / (tipStart_ - noseStart_);
                diameter_[i] = math::moveTowards(diameter, targetDiameter, slowReturn * amount, 2 * amount);
            }
            if (lastObstruction_ > -1 && newLastObstruction == -1 && noseA_[0] < 0.05)
            {
                addTransient(lastObstruction_);
            }
            lastObstruction_ = newLastObstruction;

            amount = deltaTime * movementSpeed_;
            noseDiameter_[0] = math::moveTowards(noseDiameter_[0], velumTarget_,
                amount * 0.25, amount * 0.1);
            noseA_[0] = noseDiameter_[0] * noseDiameter_[0];
        }

        void calculateReflections()
        {
            for (auto i = 0; i < n_; i++)
            {
                A_[i] = diameter_[i] * diameter_[i]; //ignoring PI etc.
            }
            for (auto i = 1; i < n_; i++)
            {
                reflection_[i] = newReflection_[i];
                if (A_[i] == 0) newReflection_[i] = 0.999; //to prevent some bad behaviour if 0
                else newReflection_[i] = (A_[i - 1] - A_[i]) / (A_[i - 1] + A_[i]);
            }

            //now at junction with nose

            reflectionLeft_ = newReflectionLeft_;
            reflectionRight_ = newReflectionRight_;
            reflectionNose_ = newReflectionNose_;
            auto sum = A_[noseStart_] + A_[noseStart_ + 1] + noseA_[0];
            newReflectionLeft_ = (2 * A_[noseStart_] - sum) / sum;
            newReflectionRight_ = (2 * A_[noseStart_ + 1] - sum) / sum;
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

    public:
        void runStep(double glottalOutput, double turbulenceNoise, double lambda)
        {
            bool updateAmplitudes = (static_cast<double>(std::rand()) / RAND_MAX < 0.1);

            //mouth
            processTransients();
            addTurbulenceNoise(turbulenceNoise);

            //_glottalReflection = -0.8 + 1.6 * Glottis.newTenseness;
            junctionOutputR_[0] = L_[0] * glottalReflection_ + glottalOutput;
            junctionOutputL_[n_] = R_[n_ - 1] * lipReflection_;

            for (auto i = 1; i < n_; i++)
            {
                auto r = reflection_[i] * (1 - lambda) + newReflection_[i] * lambda;
                auto w = r * (R_[i - 1] + L_[i]);
                junctionOutputR_[i] = R_[i - 1] - w;
                junctionOutputL_[i] = L_[i] + w;
            }

            //now at junction with nose
            auto i = noseStart_;
            auto r = newReflectionLeft_ * (1 - lambda) + reflectionLeft_ * lambda;
            junctionOutputL_[i] = r * R_[i - 1] + (1 + r) * (noseL_[0] + L_[i]);
            r = newReflectionRight_ * (1 - lambda) + reflectionRight_ * lambda;
            junctionOutputR_[i] = r * L_[i] + (1 + r) * (R_[i - 1] + noseL_[0]);
            r = newReflectionNose_ * (1 - lambda) + reflectionNose_ * lambda;
            noseJunctionOutputR_[0] = r * noseL_[0] + (1 + r) * (L_[i] + R_[i - 1]);

            for (auto i = 0; i < n_; i++)
            {
                R_[i] = junctionOutputR_[i] * 0.999;
                L_[i] = junctionOutputL_[i + 1] * 0.999;

                //_R[i] = std::clamp(_junctionOutputR[i] * _fade, -1, 1);
                //_L[i] = std::clamp(_junctionOutputL[i+1] * _fade, -1, 1);    

                if (updateAmplitudes)
                {
                    auto amplitude = std::abs(R_[i] + L_[i]);
                    if (amplitude > maxAmplitude_[i]) maxAmplitude_[i] = amplitude;
                    else maxAmplitude_[i] *= 0.999;
                }
            }

            lipOutput_ = R_[n_ - 1];

            //nose     
            noseJunctionOutputL_[noseLength_] = noseR_[noseLength_ - 1] * lipReflection_;

            for (auto i = 1; i < noseLength_; i++)
            {
                auto w = noseReflection_[i] * (noseR_[i - 1] + noseL_[i]);
                noseJunctionOutputR_[i] = noseR_[i - 1] - w;
                noseJunctionOutputL_[i] = noseL_[i] + w;
            }

            for (auto i = 0; i < noseLength_; i++)
            {
                noseR_[i] = noseJunctionOutputR_[i] * fade_;
                noseL_[i] = noseJunctionOutputL_[i + 1] * fade_;

                //_noseR[i] = std::clamp(_noseJunctionOutputR[i] * _fade, -1, 1);
                //_noseL[i] = std::clamp(_noseJunctionOutputL[i+1] * _fade, -1, 1);    

                if (updateAmplitudes)
                {
                    auto amplitude = std::abs(noseR_[i] + noseL_[i]);
                    if (amplitude > noseMaxAmplitude_[i]) noseMaxAmplitude_[i] = amplitude;
                    else noseMaxAmplitude_[i] *= 0.999;
                }
            }

            noseOutput_ = noseR_[noseLength_ - 1];
        }

        void finishBlock(double blockTime)
        {
            reshapeTract(blockTime);
            calculateReflections();
        }

    private:
        void addTransient(int position)
        {
            Transient trans{};
            trans.position = position;
            trans.timeAlive = 0;
            trans.lifeTime = 0.2;
            trans.strength = 0.3;
            trans.exponent = 200;
            transients_.push_back(trans);
        }

        void processTransients()
        {
            for (auto it = transients_.begin(); it != transients_.end(); ++it)
            {
                auto& trans = *it;
                double amplitude = trans.strength * std::pow(2, -trans.exponent * trans.timeAlive);
                R_[trans.position] += amplitude / 2;
                L_[trans.position] += amplitude / 2;
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

        void addTurbulenceNoise(double turbulenceNoise)
        {
#define TURBULENCE_FROM_DIAMETERS
#ifdef TURBULENCE_FROM_DIAMETERS
            auto pos = std::min_element(this->diameter_.begin(), this->diameter_.end());
            turbulenceIntensity_ = 1.0;
            turbulenceIndex_ = pos - this->diameter_.begin();
            turbulenceDiameter_ = *pos + 0.3;
#endif
            if (turbulenceIntensity_ == 0) return;
            addTurbulenceNoiseAtIndex(0.66 * turbulenceNoise * turbulenceIntensity_, turbulenceIndex_, turbulenceDiameter_);
        }

        void addTurbulenceNoiseAtIndex(double turbulenceNoise, double index, double diameter)
        {
            int i = static_cast<int>(index);
            double delta = index - i;
            turbulenceNoise *= glottis_.getNoiseModulator();
            double thinness0 = math::clamp(8 * (0.7 - diameter), 0., 1.);
            double openness = math::clamp(30 * (diameter - 0.3), 0., 1.);
            double noise0 = turbulenceNoise * (1 - delta) * thinness0 * openness;
            double noise1 = turbulenceNoise * delta * thinness0 * openness;
            if (i + 1 < R_.size() && i + 1 < L_.size())
            {
                R_[i + 1] += noise0 / 2;
                L_[i + 1] += noise0 / 2;
            }
            if (i + 1 < R_.size() && i + 2 < L_.size())
            {
                R_[i + 2] += noise1 / 2;
                L_[i + 2] += noise1 / 2;
            }
        }

    private:
        const Glottis& glottis_;
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

        double turbulenceIntensity_ = 0.;
        double turbulenceIndex_ = 0.;
        double turbulenceDiameter_ = 0.;
    };

    class AudioSystem
    {
    public:
        AudioSystem(Glottis& glottis, Tract& tract) : glottis_{ glottis }, tract_{ tract }, started_{ false }
        {
            blockTime_ = static_cast<double>(blockLength_) / sampleRate;
        }

        void process(double lambda, size_t len, double* out)
        {
            double* outArray = out;
            std::vector<double> inputArray1(len);
            std::vector<double> inputArray2(len);
            for (size_t j = 0; j < len; j++)
            {
                // TODO Apply 500Hz bandpass for aspirate
                inputArray1[j] = (static_cast<double>(std::rand()) / RAND_MAX) * 2. - 1.;
                // TODO Apply 1000Hz bandpass for fricative
                inputArray2[j] = (static_cast<double>(std::rand()) / RAND_MAX) * 2. - 1.;
            }
            for (size_t j = 0; j < len; j++)
            {
                double lambda1 = static_cast<double>(blockIndex_) / blockLength_;
                double lambda2 = (blockIndex_ + 0.5) / blockLength_;
                double glottalOutput = glottis_.runStep(lambda1, inputArray1[j]);

                double vocalOutput = 0;
                // Tract runs at twice the sample rate 
                tract_.runStep(glottalOutput, inputArray2[j], lambda1);
                vocalOutput += tract_.lipOutput() + tract_.noseOutput();
                tract_.runStep(glottalOutput, inputArray2[j], lambda2);
                vocalOutput += tract_.lipOutput() + tract_.noseOutput();
                outArray[j] = vocalOutput * 0.125;

                ++blockIndex_;
                if (blockIndex_ >= blockLength_)
                {
                    blockIndex_ = 0;
                    glottis_.finishBlock();
                    tract_.finishBlock(blockTime_);
                    dumpParameters();
                }
            }
        }

        void startSound()
        {
            ofs.open("dump.bin", std::ios::binary);
        }

        void started(bool value) { started_ = value; }
        bool started() const { return started_; }

    private:
        inline void push_double_to_vector(std::vector<float>& vector, double value)
        {
            vector.push_back(static_cast<float>(value));
        }

        void dumpParameters()
        {
#if 0
            std::cout << glottis_.isTouched() << " "
                << glottis_.UIFrequency() << " "
                << glottis_.UITenseness() << " "
                << glottis_.loudness() << " "
                << tract_.velumTarget() << " "
                << std::endl;
#endif
            std::vector<float> data;
            push_double_to_vector(data, 0.0);
            push_double_to_vector(data, glottis_.isTouched());
            push_double_to_vector(data, glottis_.UIFrequency());
            push_double_to_vector(data, glottis_.UITenseness());
            push_double_to_vector(data, glottis_.loudness());
            push_double_to_vector(data, tract_.velumTarget());
            auto& targetDiameter = tract_.targetDiameter();
            for (auto it = targetDiameter.begin(); it != targetDiameter.end(); ++it)
            {
                push_double_to_vector(data, *it);
            }
            data[0] = static_cast<float>(data.size());
            ofs.write(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
        }

    private:
        Glottis& glottis_;
        Tract& tract_;
        int blockIndex_ = 0;
        int blockLength_ = 512;
        double blockTime_ = 1.;
        bool started_ = false;
        bool soundOn_ = false;
        std::ofstream ofs;
    };

    class TractUI {
    public:
        TractUI(Tract& tract) : tract_{ tract }
        {
            setRestDiameter();
            tract_.targetDiameter() = tract_.restDiameter();
        }

#ifdef USE_SDL
        double getIndex(int x, int y) const
        {
            return static_cast<double>(tract_.n() * (x - tractRect_.x)) / tractRect_.w;
        }

        double getDiameter(int x, int y) const
        {
            double v = static_cast<double>(y - (tractRect_.y + (tractRect_.h - 10) / 2 + 10));
            return static_cast<double>(v) * (tract_.n() * stepLength_) / tractRect_.w;
        }

        void drawPitchControl(SDL_Renderer* renderer)
        {

        }
#endif

        void setRestDiameter()
        {
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

        void handleTouches(std::map<std::string, Touch>& touchesWithMouse)
        {

            double tongueIndexCentre = 0.5 * (tongueLowerIndexBound_ + tongueUpperIndexBound_);

            if (!tongueTouch_.empty() && !touchesWithMouse[tongueTouch_].alive) tongueTouch_ = "";

            if (tongueTouch_.empty()) {
                for (auto it = touchesWithMouse.begin(); it != touchesWithMouse.end(); ++it)
                {
                    auto& touch = it->second;
                    if (!touch.alive) continue;
                    if (touch.fricative_intensity == 1.) continue; //only new touches will pass this
                    double index = touch.index;
                    double diameter = touch.diameter;
                    if (index >= tongueLowerIndexBound_ - 4 && index <= tongueUpperIndexBound_ + 4
                        && diameter >= innerTongueControlRadius_ - 0.5 && diameter <= outerTongueControlRadius_ + 0.5)
                    {
                        tongueTouch_ = touch.id;
                    }
                }
            }

            if (!tongueTouch_.empty())
            {
                auto& touch = touchesWithMouse.at(tongueTouch_);
                double index = touch.index;
                double diameter = touch.diameter;
                double fromPoint = (outerTongueControlRadius_ - diameter) / (outerTongueControlRadius_ - innerTongueControlRadius_);
                fromPoint = math::clamp(fromPoint, 0., 1.);
                fromPoint = std::pow(fromPoint, 0.58) - 0.2 * (fromPoint * fromPoint - fromPoint); //horrible kludge to fit curve to straight line
                tongueDiameter_ = math::clamp(diameter, innerTongueControlRadius_, outerTongueControlRadius_);
                //tongueIndex = Math.clamp(index, tongueLowerIndexBound, tongueUpperIndexBound);
                double out = fromPoint * 0.5 * (tongueUpperIndexBound_ - tongueLowerIndexBound_);
                tongueIndex_ = math::clamp(index, tongueIndexCentre - out, tongueIndexCentre + out);
            }

            setRestDiameter();
            auto& targetDiameter = tract_.targetDiameter();
            std::copy(tract_.restDiameter().begin(), tract_.restDiameter().end(), targetDiameter.begin());

            //other constrictions and nose
            tract_.velumTarget(0.01);
            for (auto it = touchesWithMouse.begin(); it != touchesWithMouse.end(); ++it)
            {
                auto& touch = it->second;
                if (!touch.alive) continue;
                double index = touch.index;
                double diameter = touch.diameter;
                if (index > tract_.noseStart() && diameter < -noseOffset_)
                {
                    tract_.velumTarget(0.4);
                }
                if (diameter < -0.85 - noseOffset_) continue;
                diameter -= 0.3;
                if (diameter < 0) diameter = 0;
                double width = 2.;
                if (index < 25) width = 10;
                else if (index >= tract_.tipStart()) width = 5;
                else width = 10 - 5 * (index - 25) / (tract_.tipStart() - 25);
                if (index >= 2 && index < tract_.n() && diameter < 3)
                {
                    int intIndex = static_cast<int>(std::round(index));
                    for (int i = -static_cast<int>(std::ceil(width)) - 1; i < width + 1; i++)
                    {
                        if (intIndex + i < 0 || intIndex + i >= tract_.n()) continue;
                        double relpos = (intIndex + i) - index;
                        relpos = std::abs(relpos) - 0.5;
                        double shrink;
                        if (relpos <= 0) shrink = 0;
                        else if (relpos > width) shrink = 1;
                        else shrink = 0.5 * (1 - std::cos(M_PI * relpos / width));
                        if (diameter < targetDiameter[intIndex + i])
                        {
                            targetDiameter[intIndex + i] = diameter + (targetDiameter[intIndex + i] - diameter) * shrink;
                        }
                    }
                }
            }

            // addTurbulenceNoise()
            for (auto it = touchesWithMouse.begin(); it != touchesWithMouse.end(); ++it)
            {
                auto& touch = it->second;
                if (touch.index<2 || touch.index>tract_.n()) continue;
                if (touch.diameter <= 0) continue;
                tract_.turbulanceNoise(touch.fricative_intensity, touch.index, touch.diameter);
            }
        }

#ifdef USE_SDL
        void draw(SDL_Renderer* renderer)
        {
            SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
            SDL_RenderClear(renderer);
            SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
            auto& diameter = tract_.diameter();
            for (size_t i = 0; i < diameter.size(); i++) {
                if (i == tract_.noseStart()) {
                    SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
                }
                else {
                    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
                }
                const double v = diameter[i] * tractRect_.w / (diameter.size() * stepLength_);
                SDL_Rect rect{
                    static_cast<int>(tractRect_.x + tractRect_.w * i / diameter.size()),
                    tractRect_.y + (tractRect_.h - 10) / 2 + 10,
                    static_cast<int>(tractRect_.w * (i + 1) / diameter.size() - tractRect_.w * i / diameter.size()),
                    static_cast<int>(v)
                };
                SDL_RenderFillRect(renderer, &rect);
            }

            auto noseDiameter = tract_.noseDiameter();
            for (int i = 0; i < noseDiameter.size(); i++) {
                SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
                const double v = noseDiameter[i] * tractRect_.w / (diameter.size() * stepLength_);
                SDL_Rect rect{
                    static_cast<int>(tractRect_.x + tractRect_.w * (tract_.noseStart() + i) / diameter.size()),
                    tractRect_.y + (tractRect_.h - 10) / 2 - static_cast<int>(v),
                    static_cast<int>(tractRect_.w * (tract_.noseStart() + i + 1) / diameter.size() - tractRect_.w * (tract_.noseStart() + i) / diameter.size()),
                    static_cast<int>(v)
                };
                SDL_RenderFillRect(renderer, &rect);
            }

            drawPitchControl(renderer);

            SDL_RenderPresent(renderer);
        }
#endif

    private:
        Tract& tract_;
        const double innerTongueControlRadius_ = 2.05;
        const double outerTongueControlRadius_ = 3.5;
        const double tongueLowerIndexBound_ = tract_.bladeStart() + 2;
        const double tongueUpperIndexBound_ = tract_.tipStart() - 3;
#ifdef USE_SDL
        SDL_Rect tractRect_{ 20, 20, 600, 200 };
#endif
        std::string tongueTouch_;
        double tongueIndex_ = (tongueLowerIndexBound_ + tongueUpperIndexBound_) / 2.;
        double tongueDiameter_ = (innerTongueControlRadius_ + outerTongueControlRadius_) / 2.;
        double gridOffset_ = 1.7;
        double noseOffset_ = 0.8;
        const double stepLength_ = 0.397;
    };

    class UI
    {
    public:
        UI(int n) : glottis_{}, tract_{ glottis_, n }, tractUI_{ tract_ }, audioSystem_{ glottis_, tract_ }
        {
        }

#ifdef USE_SDL
        void draw(SDL_Renderer* renderer)
        {
        }

        void startMouse(const SDL_MouseButtonEvent* event)
        {
            if (!audioSystem_.started())
            {
                audioSystem_.started(true);
                audioSystem_.startSound();
            }

            Touch touch = {};
            touch.startTime = std::clock() / static_cast<double>(CLOCKS_PER_SEC);
            touch.fricative_intensity = 0.;
            touch.endTime = 0.;
            touch.alive = true;
            touch.id = "mouse" + std::to_string(std::rand());
            touch.x = event->x;
            touch.y = event->y;
            touch.index = tractUI_.getIndex(event->x, event->y);
            touch.diameter = tractUI_.getDiameter(event->x, event->y);
            mouseTouch_ = touch.id;
            touchesWithMouse_[touch.id] = touch;
            handleTouches();
        }

        void moveMouse(const SDL_MouseMotionEvent* event)
        {
            if (mouseTouch_.empty()) return;
            auto& touch = touchesWithMouse_.at(mouseTouch_);
            if (!touch.alive) return;
            touch.x = event->x;
            touch.y = event->y;
            touch.index = tractUI_.getIndex(event->x, event->y);
            touch.diameter = tractUI_.getDiameter(event->x, event->y);
            handleTouches();
        }

        void endMouse(const SDL_MouseButtonEvent* event)
        {
            if (mouseTouch_.empty()) return;
            double time = std::clock() / static_cast<double>(CLOCKS_PER_SEC);
            auto& touch = touchesWithMouse_.at(mouseTouch_);
            mouseTouch_ = "";
            if (!touch.alive) return;
            touch.alive = false;
            touch.endTime = time;
            handleTouches();
        }
#endif

        void handleTouches()
        {
            tractUI_.handleTouches(touchesWithMouse_);
            glottis_.handleTouches(touchesWithMouse_);
        }

        void updateTouches()
        {
            double time = std::clock() / static_cast<double>(CLOCKS_PER_SEC);
            double fricativeAttackTime = 0.1;
            for (auto it = touchesWithMouse_.begin(); it != touchesWithMouse_.end();)
            {
                auto& touch = it->second;
                if (!(touch.alive) && (time > touch.endTime + 1))
                {
                    it = touchesWithMouse_.erase(it);
                }
                else if (touch.alive)
                {
                    ++it;
                    touch.fricative_intensity = math::clamp((time - touch.startTime) / fricativeAttackTime, 0., 1.);
                }
                else
                {
                    ++it;
                    touch.fricative_intensity = math::clamp(1 - (time - touch.endTime) / fricativeAttackTime, 0., 1.);
                }
            }
        }
        Glottis& glottis() { return glottis_; }
        Tract& tract() { return tract_; }
        TractUI& tractUI() { return tractUI_; }
        AudioSystem& audioSystem() { return audioSystem_; }

    private:
        Glottis glottis_;
        Tract tract_;
        TractUI tractUI_;
        AudioSystem audioSystem_;
        std::map<std::string, Touch> touchesWithMouse_;
        std::string mouseTouch_;
    };
}
