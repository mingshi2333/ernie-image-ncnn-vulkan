// Isolated harness for the pinned reference project's TextEncoder tokenization.
#include "bpe_tokenizer.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>

int main(int argc, char** argv)
{
    if (argc < 4) return 2;
    SpecialTokensConfig spec;
    spec.bos_token = "<s>";
    spec.eos_token = "</s>";
    spec.unk_token = "<unk>";
    spec.pad_token = "<pad>";
    auto tokenizer = BpeTokenizer::LoadFromFiles(argv[1], argv[2], spec, false, true, true);
    tokenizer.SetByteLevelRegexPretokenizer(true);
    tokenizer.SetIgnoreMerges(true);
    for (int i = 3; i < argc; ++i)
    {
        std::ifstream stream(argv[i], std::ios::binary);
        if (!stream) return 3;
        std::string prompt((std::istreambuf_iterator<char>(stream)), {});
        auto ids = tokenizer.encode(prompt, true, false, false, false);
        if (ids.empty()) ids.push_back(1);
        if (ids.size() > 2048) ids.resize(2048);
        std::cout << '[';
        for (size_t j = 0; j < ids.size(); ++j)
            std::cout << (j ? "," : "") << ids[j];
        std::cout << "]\n";
    }
    return 0;
}
