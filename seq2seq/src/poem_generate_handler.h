#ifndef POEM_GENERATE_HANDLER_H_INCLUDED
#define POEM_GENERATE_HANDLER_H_INCLUDED
#include <random>
#include <fstream>

#include "poem_generate.h"
#include <boost/program_options.hpp>

struct PoemGeneratorHandler
{
    PoemGenerator &pg;
    std::mt19937 rng;
    PoemGeneratorHandler(PoemGenerator &pg , size_t seed=1314);
    ~PoemGeneratorHandler();

    void read_train_data(std::ifstream &is , std::vector<Poem> &poems);
    
    void finish_reading_training_data();
    void finish_reading_training_data(boost::program_options::variables_map &var_map);

    void build_model();

    void train(const std::vector<Poem> &poems , size_t max_epoch , size_t report_freq=1000);
    void generate(const std::string &first_seq, std::vector<std::string> &generated_poem);

    void save_model(std::ofstream &os);
    void load_model(std::ifstream &is);

    // tools 
    void slice_utf8_sents2single_words(const std::string &usent, std::vector<std::string> &words_cont);
};


#endif
