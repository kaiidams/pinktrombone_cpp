#include <vector>
#include <fstream>
#include <iostream>
#include <cassert>
#include "capi.h"

#if 0
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
#endif

int main(int argc, char *argv[])
{
    std::ifstream ifs("dump.bin", std::ios_base::binary);
    std::ofstream ofs("audio.bin", std::ios_base::binary);
    PinkTrombone* pt = PinkTrombone_new(44);

    assert(ifs.is_open());
    assert(ofs.is_open());

    // setRestDiameter(tract_);
    // tract_.targetDiameter() = tract_.restDiameter();

    int i = 0;
    while (!ifs.eof())
    {
        int ret;

        std::vector<float> sdata(50);
        std::vector<double> ddata(50);

        ifs.read(reinterpret_cast<char*>(sdata.data()), sdata.size() * sizeof (float));
        // std::cout << data[0] << std::endl;
        std::copy(sdata.begin(), sdata.end(), ddata.begin());
        // std::cout << ddata[2] << std::endl;
        ret = PinkTrombone_control(pt, ddata.data() + 1, ddata.size() - 1);
        assert(ret == 0);

        std::vector<double> out(512);
        ret = PinkTrombone_process(pt, out.data(), out.size());
        assert(ret == 0);

        ofs.write(reinterpret_cast<char*>(out.data()), out.size() * sizeof (double));

        ++i;
        std::cout << i << "\r";
    }
    std::cout << std::endl;
}
