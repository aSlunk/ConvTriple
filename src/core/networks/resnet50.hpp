#ifndef RESNET50_HPP_
#define RESNET50_HPP_

#include <vector>

#include "defs.hpp"

namespace ResNet50 {
using std::vector;

vector<gemini::HomConv2DSS::Meta> init_layers_conv_cheetah();
vector<gemini::HomConv2DSS::Meta> init_layers_conv_alt();

vector<gemini::HomBNSS::Meta> init_layers_bn_cheetah();
vector<gemini::HomBNSS::Meta> init_layers_bn_alt();
} // namespace ResNet50

vector<gemini::HomBNSS::Meta> ResNet50::init_layers_bn_alt() {
    vector<gemini::HomBNSS::Meta> layers;
    layers.reserve(53);
    layers.push_back(Utils::init_meta_bn(65, 12544));
    layers.push_back(Utils::init_meta_bn(64, 3136));
    layers.push_back(Utils::init_meta_bn(64, 3136));
    layers.push_back(Utils::init_meta_bn(256, 3136));
    layers.push_back(Utils::init_meta_bn(256, 3136));
    layers.push_back(Utils::init_meta_bn(64, 3136));
    layers.push_back(Utils::init_meta_bn(64, 3136));
    layers.push_back(Utils::init_meta_bn(256, 3136));
    layers.push_back(Utils::init_meta_bn(64, 3136));
    layers.push_back(Utils::init_meta_bn(64, 3136));
    layers.push_back(Utils::init_meta_bn(256, 3136));
    layers.push_back(Utils::init_meta_bn(128, 3136));
    layers.push_back(Utils::init_meta_bn(128, 784));
    layers.push_back(Utils::init_meta_bn(512, 784));
    layers.push_back(Utils::init_meta_bn(512, 784));
    layers.push_back(Utils::init_meta_bn(128, 784));
    layers.push_back(Utils::init_meta_bn(128, 784));
    layers.push_back(Utils::init_meta_bn(512, 784));
    layers.push_back(Utils::init_meta_bn(128, 784));
    layers.push_back(Utils::init_meta_bn(128, 784));
    layers.push_back(Utils::init_meta_bn(512, 784));
    layers.push_back(Utils::init_meta_bn(128, 784));
    layers.push_back(Utils::init_meta_bn(128, 784));
    layers.push_back(Utils::init_meta_bn(512, 784));
    layers.push_back(Utils::init_meta_bn(256, 784));
    layers.push_back(Utils::init_meta_bn(256, 196));
    layers.push_back(Utils::init_meta_bn(1024, 196));
    layers.push_back(Utils::init_meta_bn(1024, 196));
    layers.push_back(Utils::init_meta_bn(256, 196));
    layers.push_back(Utils::init_meta_bn(256, 196));
    layers.push_back(Utils::init_meta_bn(1024, 196));
    layers.push_back(Utils::init_meta_bn(256, 196));
    layers.push_back(Utils::init_meta_bn(256, 196));
    layers.push_back(Utils::init_meta_bn(1024, 196));
    layers.push_back(Utils::init_meta_bn(256, 196));
    layers.push_back(Utils::init_meta_bn(256, 196));
    layers.push_back(Utils::init_meta_bn(1024, 196));
    layers.push_back(Utils::init_meta_bn(256, 196));
    layers.push_back(Utils::init_meta_bn(256, 196));
    layers.push_back(Utils::init_meta_bn(1024, 196));
    layers.push_back(Utils::init_meta_bn(256, 196));
    layers.push_back(Utils::init_meta_bn(256, 196));
    layers.push_back(Utils::init_meta_bn(1024, 196));
    layers.push_back(Utils::init_meta_bn(512, 196));
    layers.push_back(Utils::init_meta_bn(512, 49));
    layers.push_back(Utils::init_meta_bn(2048, 49));
    layers.push_back(Utils::init_meta_bn(2048, 49));
    layers.push_back(Utils::init_meta_bn(512, 49));
    layers.push_back(Utils::init_meta_bn(512, 49));
    layers.push_back(Utils::init_meta_bn(2048, 49));
    layers.push_back(Utils::init_meta_bn(512, 49));
    layers.push_back(Utils::init_meta_bn(512, 49));
    layers.push_back(Utils::init_meta_bn(2048, 49));
    std::cerr << layers.size() << "\n";
    return layers;
}

vector<gemini::HomBNSS::Meta> ResNet50::init_layers_bn_cheetah() {
    vector<gemini::HomBNSS::Meta> layers;
    layers.reserve(49);
    layers.push_back(Utils::init_meta_bn(64, 3136));
    layers.push_back(Utils::init_meta_bn(64, 3136));
    layers.push_back(Utils::init_meta_bn(64, 3136));
    layers.push_back(Utils::init_meta_bn(256, 3136));
    layers.push_back(Utils::init_meta_bn(64, 3136));
    layers.push_back(Utils::init_meta_bn(64, 3136));
    layers.push_back(Utils::init_meta_bn(256, 3136));
    layers.push_back(Utils::init_meta_bn(64, 3136));
    layers.push_back(Utils::init_meta_bn(64, 3136));
    layers.push_back(Utils::init_meta_bn(256, 3136));
    layers.push_back(Utils::init_meta_bn(128, 3136));
    layers.push_back(Utils::init_meta_bn(128, 784));
    layers.push_back(Utils::init_meta_bn(512, 784));
    layers.push_back(Utils::init_meta_bn(128, 784));
    layers.push_back(Utils::init_meta_bn(128, 784));
    layers.push_back(Utils::init_meta_bn(512, 784));
    layers.push_back(Utils::init_meta_bn(128, 784));
    layers.push_back(Utils::init_meta_bn(128, 784));
    layers.push_back(Utils::init_meta_bn(512, 784));
    layers.push_back(Utils::init_meta_bn(128, 784));
    layers.push_back(Utils::init_meta_bn(128, 784));
    layers.push_back(Utils::init_meta_bn(512, 784));
    layers.push_back(Utils::init_meta_bn(256, 784));
    layers.push_back(Utils::init_meta_bn(256, 196));
    layers.push_back(Utils::init_meta_bn(1024, 196));
    layers.push_back(Utils::init_meta_bn(256, 196));
    layers.push_back(Utils::init_meta_bn(256, 196));
    layers.push_back(Utils::init_meta_bn(1024, 196));
    layers.push_back(Utils::init_meta_bn(256, 196));
    layers.push_back(Utils::init_meta_bn(256, 196));
    layers.push_back(Utils::init_meta_bn(1024, 196));
    layers.push_back(Utils::init_meta_bn(256, 196));
    layers.push_back(Utils::init_meta_bn(256, 196));
    layers.push_back(Utils::init_meta_bn(1024, 196));
    layers.push_back(Utils::init_meta_bn(256, 196));
    layers.push_back(Utils::init_meta_bn(256, 196));
    layers.push_back(Utils::init_meta_bn(1024, 196));
    layers.push_back(Utils::init_meta_bn(256, 196));
    layers.push_back(Utils::init_meta_bn(256, 196));
    layers.push_back(Utils::init_meta_bn(1024, 196));
    layers.push_back(Utils::init_meta_bn(512, 196));
    layers.push_back(Utils::init_meta_bn(512, 49));
    layers.push_back(Utils::init_meta_bn(2048, 49));
    layers.push_back(Utils::init_meta_bn(512, 49));
    layers.push_back(Utils::init_meta_bn(512, 49));
    layers.push_back(Utils::init_meta_bn(2048, 49));
    layers.push_back(Utils::init_meta_bn(512, 49));
    layers.push_back(Utils::init_meta_bn(512, 49));
    layers.push_back(Utils::init_meta_bn(2048, 49));
    std::cerr << layers.size() << "\n";
    return layers;
}

vector<gemini::HomConv2DSS::Meta> ResNet50::init_layers_conv_alt() {
    std::vector<gemini::HomConv2DSS::Meta> layers;
    layers.reserve(53);
    layers.push_back(Utils::init_meta_conv(3, 224, 224, 3, 7, 7, 64, 2, 3));
    layers.push_back(Utils::init_meta_conv(64, 56, 56, 64, 1, 1, 64, 1, 0));       // L1
    layers.push_back(Utils::init_meta_conv(64, 56, 56, 64, 3, 3, 64, 1, 1));       // L2
    layers.push_back(Utils::init_meta_conv(64, 56, 56, 64, 1, 1, 256, 1, 0));      // L3
    layers.push_back(Utils::init_meta_conv(64, 56, 56, 64, 1, 1, 256, 1, 0));      // L4
    layers.push_back(Utils::init_meta_conv(256, 56, 56, 256, 1, 1, 64, 1, 0));     // L5
    layers.push_back(Utils::init_meta_conv(64, 56, 56, 64, 3, 3, 64, 1, 1));       // L6
    layers.push_back(Utils::init_meta_conv(64, 56, 56, 64, 1, 1, 256, 1, 0));      // L7
    layers.push_back(Utils::init_meta_conv(256, 56, 56, 256, 1, 1, 64, 1, 0));     // L8
    layers.push_back(Utils::init_meta_conv(64, 56, 56, 64, 3, 3, 64, 1, 1));       // L9
    layers.push_back(Utils::init_meta_conv(64, 56, 56, 64, 1, 1, 256, 1, 1));      // L10
    layers.push_back(Utils::init_meta_conv(256, 56, 56, 256, 1, 1, 128, 1, 0));    // L11
    layers.push_back(Utils::init_meta_conv(128, 56, 56, 128, 3, 3, 128, 2, 1));    // L12
    layers.push_back(Utils::init_meta_conv(128, 28, 28, 128, 1, 1, 512, 1, 0));    // L13
    layers.push_back(Utils::init_meta_conv(256, 56, 56, 256, 1, 1, 512, 2, 0));    // L14
    layers.push_back(Utils::init_meta_conv(512, 28, 28, 512, 1, 1, 128, 1, 0));    // L15
    layers.push_back(Utils::init_meta_conv(128, 28, 28, 128, 3, 3, 128, 1, 1));    // L16
    layers.push_back(Utils::init_meta_conv(128, 28, 28, 128, 1, 1, 512, 1, 0));    // L17
    layers.push_back(Utils::init_meta_conv(512, 28, 28, 512, 1, 1, 128, 1, 0));    // L18
    layers.push_back(Utils::init_meta_conv(128, 28, 28, 128, 3, 3, 128, 1, 1));    // L19
    layers.push_back(Utils::init_meta_conv(128, 28, 28, 128, 1, 1, 512, 1, 0));    // L20
    layers.push_back(Utils::init_meta_conv(512, 28, 28, 512, 1, 1, 128, 1, 0));    // L21
    layers.push_back(Utils::init_meta_conv(128, 28, 28, 128, 3, 3, 128, 1, 1));    // L22
    layers.push_back(Utils::init_meta_conv(128, 28, 28, 128, 1, 1, 512, 1, 0));    // L23
    layers.push_back(Utils::init_meta_conv(512, 28, 28, 512, 1, 1, 256, 1, 0));    // L24
    layers.push_back(Utils::init_meta_conv(256, 28, 28, 256, 3, 3, 256, 2, 1));    // L25
    layers.push_back(Utils::init_meta_conv(256, 14, 14, 256, 1, 1, 1024, 1, 0));   // L26
    layers.push_back(Utils::init_meta_conv(512, 28, 28, 512, 1, 1, 1024, 2, 0));   // L27
    layers.push_back(Utils::init_meta_conv(1024, 14, 14, 1024, 1, 1, 256, 1, 0));  // L28
    layers.push_back(Utils::init_meta_conv(256, 14, 14, 256, 3, 3, 256, 1, 1));    // L29
    layers.push_back(Utils::init_meta_conv(256, 14, 14, 256, 1, 1, 1024, 1, 0));   // L30
    layers.push_back(Utils::init_meta_conv(1024, 14, 14, 1024, 1, 1, 256, 1, 0));  // L31
    layers.push_back(Utils::init_meta_conv(256, 14, 14, 256, 3, 3, 256, 1, 1));    // L32
    layers.push_back(Utils::init_meta_conv(256, 14, 14, 256, 1, 1, 1024, 1, 0));   // L33
    layers.push_back(Utils::init_meta_conv(1024, 14, 14, 1024, 1, 1, 256, 1, 0));  // L34
    layers.push_back(Utils::init_meta_conv(256, 14, 14, 256, 3, 3, 256, 1, 1));    // L35
    layers.push_back(Utils::init_meta_conv(256, 14, 14, 256, 1, 1, 1024, 1, 0));   // L36
    layers.push_back(Utils::init_meta_conv(1024, 14, 14, 1024, 1, 1, 256, 1, 0));  // L37
    layers.push_back(Utils::init_meta_conv(256, 14, 14, 256, 3, 3, 256, 1, 1));    // L38
    layers.push_back(Utils::init_meta_conv(256, 14, 14, 256, 1, 1, 1024, 1, 0));   // L39
    layers.push_back(Utils::init_meta_conv(1024, 14, 14, 1024, 1, 1, 256, 1, 0));  // L40
    layers.push_back(Utils::init_meta_conv(256, 14, 14, 256, 3, 3, 256, 1, 1));    // L41
    layers.push_back(Utils::init_meta_conv(256, 14, 14, 256, 1, 1, 1024, 1, 0));   // L42
    layers.push_back(Utils::init_meta_conv(1024, 14, 14, 1024, 1, 1, 512, 1, 0));  // L43
    layers.push_back(Utils::init_meta_conv(512, 14, 14, 512, 3, 3, 512, 2, 1));    // L44
    layers.push_back(Utils::init_meta_conv(512, 7, 7, 512, 1, 1, 2048, 1, 0));     // L45
    layers.push_back(Utils::init_meta_conv(1024, 14, 14, 1024, 1, 1, 2048, 2, 0)); // L46
    layers.push_back(Utils::init_meta_conv(2048, 7, 7, 2048, 1, 1, 512, 1, 0));    // L47
    layers.push_back(Utils::init_meta_conv(512, 7, 7, 512, 3, 3, 512, 1, 1));      // L48
    layers.push_back(Utils::init_meta_conv(512, 7, 7, 512, 1, 1, 2048, 1, 0));     // L49
    layers.push_back(Utils::init_meta_conv(2048, 7, 7, 2048, 1, 1, 512, 1, 0));    // L50
    layers.push_back(Utils::init_meta_conv(512, 7, 7, 512, 3, 3, 512, 1, 1));      // L51
    layers.push_back(Utils::init_meta_conv(512, 7, 7, 512, 1, 1, 2048, 1, 0));     // L52
    return layers;
}

vector<gemini::HomConv2DSS::Meta> ResNet50::init_layers_conv_cheetah() {
    vector<gemini::HomConv2DSS::Meta> layers;
    layers.reserve(53);
    layers.push_back(Utils::init_meta_conv(3, 230, 230, 3, 7, 7, 64, 2, 0));
    layers.push_back(Utils::init_meta_conv(64, 56, 56, 64, 1, 1, 256, 1, 0));
    layers.push_back(Utils::init_meta_conv(64, 56, 56, 64, 1, 1, 64, 1, 0));
    layers.push_back(Utils::init_meta_conv(64, 56, 56, 64, 3, 3, 64, 1, 1));
    layers.push_back(Utils::init_meta_conv(64, 56, 56, 64, 1, 1, 256, 1, 0));
    layers.push_back(Utils::init_meta_conv(256, 56, 56, 256, 1, 1, 64, 1, 0));
    layers.push_back(Utils::init_meta_conv(64, 56, 56, 64, 3, 3, 64, 1, 1));
    layers.push_back(Utils::init_meta_conv(64, 56, 56, 64, 1, 1, 256, 1, 0));
    layers.push_back(Utils::init_meta_conv(256, 56, 56, 256, 1, 1, 64, 1, 0));
    layers.push_back(Utils::init_meta_conv(64, 56, 56, 64, 3, 3, 64, 1, 1));
    layers.push_back(Utils::init_meta_conv(64, 56, 56, 64, 1, 1, 256, 1, 0));
    layers.push_back(Utils::init_meta_conv(256, 56, 56, 256, 1, 1, 512, 2, 0));
    layers.push_back(Utils::init_meta_conv(256, 56, 56, 256, 1, 1, 128, 1, 0));
    layers.push_back(Utils::init_meta_conv(128, 58, 58, 128, 3, 3, 128, 2, 0));
    layers.push_back(Utils::init_meta_conv(128, 28, 28, 128, 1, 1, 512, 1, 0));
    layers.push_back(Utils::init_meta_conv(512, 28, 28, 512, 1, 1, 128, 1, 0));
    layers.push_back(Utils::init_meta_conv(128, 28, 28, 128, 3, 3, 128, 1, 1));
    layers.push_back(Utils::init_meta_conv(128, 28, 28, 128, 1, 1, 512, 1, 0));
    layers.push_back(Utils::init_meta_conv(512, 28, 28, 512, 1, 1, 128, 1, 0));
    layers.push_back(Utils::init_meta_conv(128, 28, 28, 128, 3, 3, 128, 1, 1));
    layers.push_back(Utils::init_meta_conv(128, 28, 28, 128, 1, 1, 512, 1, 0));
    layers.push_back(Utils::init_meta_conv(512, 28, 28, 512, 1, 1, 128, 1, 0));
    layers.push_back(Utils::init_meta_conv(128, 28, 28, 128, 3, 3, 128, 1, 1));
    layers.push_back(Utils::init_meta_conv(128, 28, 28, 128, 1, 1, 512, 1, 0));
    layers.push_back(Utils::init_meta_conv(512, 28, 28, 512, 1, 1, 1024, 2, 0));
    layers.push_back(Utils::init_meta_conv(512, 28, 28, 512, 1, 1, 256, 1, 0));
    layers.push_back(Utils::init_meta_conv(256, 30, 30, 256, 3, 3, 256, 2, 0));
    layers.push_back(Utils::init_meta_conv(256, 14, 14, 256, 1, 1, 1024, 1, 0));
    layers.push_back(Utils::init_meta_conv(1024, 14, 14, 1024, 1, 1, 256, 1, 0));
    layers.push_back(Utils::init_meta_conv(256, 14, 14, 256, 3, 3, 256, 1, 1));
    layers.push_back(Utils::init_meta_conv(256, 14, 14, 256, 1, 1, 1024, 1, 0));
    layers.push_back(Utils::init_meta_conv(1024, 14, 14, 1024, 1, 1, 256, 1, 0));
    layers.push_back(Utils::init_meta_conv(256, 14, 14, 256, 3, 3, 256, 1, 1));
    layers.push_back(Utils::init_meta_conv(256, 14, 14, 256, 1, 1, 1024, 1, 0));
    layers.push_back(Utils::init_meta_conv(1024, 14, 14, 1024, 1, 1, 256, 1, 0));
    layers.push_back(Utils::init_meta_conv(256, 14, 14, 256, 3, 3, 256, 1, 1));
    layers.push_back(Utils::init_meta_conv(256, 14, 14, 256, 1, 1, 1024, 1, 0));
    layers.push_back(Utils::init_meta_conv(1024, 14, 14, 1024, 1, 1, 256, 1, 0));
    layers.push_back(Utils::init_meta_conv(256, 14, 14, 256, 3, 3, 256, 1, 1));
    layers.push_back(Utils::init_meta_conv(256, 14, 14, 256, 1, 1, 1024, 1, 0));
    layers.push_back(Utils::init_meta_conv(1024, 14, 14, 1024, 1, 1, 256, 1, 0));
    layers.push_back(Utils::init_meta_conv(256, 14, 14, 256, 3, 3, 256, 1, 1));
    layers.push_back(Utils::init_meta_conv(256, 14, 14, 256, 1, 1, 1024, 1, 0));
    layers.push_back(Utils::init_meta_conv(1024, 14, 14, 1024, 1, 1, 2048, 2, 0));
    layers.push_back(Utils::init_meta_conv(1024, 14, 14, 1024, 1, 1, 512, 1, 0));
    layers.push_back(Utils::init_meta_conv(512, 16, 16, 512, 3, 3, 512, 2, 0));
    layers.push_back(Utils::init_meta_conv(512, 7, 7, 512, 1, 1, 2048, 1, 0));
    layers.push_back(Utils::init_meta_conv(2048, 7, 7, 2048, 1, 1, 512, 1, 0));
    layers.push_back(Utils::init_meta_conv(512, 7, 7, 512, 3, 3, 512, 1, 1));
    layers.push_back(Utils::init_meta_conv(512, 7, 7, 512, 1, 1, 2048, 1, 0));
    layers.push_back(Utils::init_meta_conv(2048, 7, 7, 2048, 1, 1, 512, 1, 0));
    layers.push_back(Utils::init_meta_conv(512, 7, 7, 512, 3, 3, 512, 1, 1));
    layers.push_back(Utils::init_meta_conv(512, 7, 7, 512, 1, 1, 2048, 1, 0));
    return layers;
}

#endif