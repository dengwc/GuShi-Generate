#ifndef POEM_GENERATE_HANDLER_H_INCLUDED
#define POEM_GENERATE_HANDLER_H_INCLUDED
#include <random>
#include <fstream>
#include <boost/log/trivial.hpp>
#include <boost/log/core.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options.hpp>

#include "poem_generate.h"
#include "timestat.hpp"
#include "thirdparty/utf8.h"


template <typename RNNType>
struct PoemGeneratorHandler
{
    PoemGenerator<RNNType> pg;
    std::mt19937 rng;
    PoemGeneratorHandler(size_t seed=1314);
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


// ---------------- Template Class Implementation -----------------

template <typename RNNType>
PoemGeneratorHandler<RNNType>::PoemGeneratorHandler(std::size_t seed)
    :pg(PoemGenerator<RNNType>()) , rng(seed)
{}

template <typename RNNType>
PoemGeneratorHandler<RNNType>::~PoemGeneratorHandler()
{}

template <typename RNNType>
void PoemGeneratorHandler<RNNType>::read_train_data(std::ifstream &is, std::vector<Poem> &poems)
{
    std::vector<Poem> tmp_poems;
    tmp_poems.reserve(0x20000); // 2^17 
    std::string line;
    while (getline(is, line))
    {
        std::vector<std::string> sent_cont;
        boost::split(sent_cont, line, boost::is_any_of("\t"));
        Poem poem(sent_cont.size());
        for (std::size_t sent_idx = 0 ; sent_idx < sent_cont.size() ; ++sent_idx)
        {
            const std::string &sent = sent_cont.at(sent_idx);
            std::vector<std::string> words_cont;
            boost::split(words_cont, sent, boost::is_any_of(" "));
            IndexSeq &cur_sent = poem.at(sent_idx);
            cur_sent.resize(words_cont.size());
            for (std::size_t word_idx = 0; word_idx < words_cont.size(); ++word_idx)
            {
                cur_sent.at(word_idx) = pg.word_dict.Convert(words_cont.at(word_idx)); // add to dict and buid poems
            }
        }
        tmp_poems.push_back(poem);
    }
    swap(tmp_poems, poems);
}

template <typename RNNType>
void PoemGeneratorHandler<RNNType>::finish_reading_training_data()
{
    assert(!pg.word_dict.is_frozen());
    pg.word_dict.Convert("EOS_OUTPUT");
    pg.word_dict.Freeze();
    pg.word_dict.SetUnk("UNK_STR");
    pg.word_dict_size = pg.word_dict.size();

    pg.word_embedding_dim = 50;
    pg.enc_h_dim = 500;
    pg.enc_stacked_layer_num = 3;
    pg.dec_h_dim = 500;
    pg.dec_stacked_layer_num = 3;

    pg.enc_hidden_layer_output_dim = static_cast<unsigned>(pg.enc_h_dim * pg.enc_stacked_layer_num * 1.5);
    pg.enc_output_layer_output_dim = pg.dec_h_dim * pg.dec_stacked_layer_num; // output will be split to `dec_stacked_layer_num` pieces to init dec hidden   

}

template <typename RNNType>
void PoemGeneratorHandler<RNNType>::finish_reading_training_data(boost::program_options::variables_map &var_map)
{
    assert(!pg.word_dict.is_frozen());
    pg.word_dict.Convert(pg.EOS_STR);
    pg.word_dict.Freeze();
    pg.word_dict.SetUnk(pg.UNK_STR);
    pg.word_dict_size = pg.word_dict.size();

    pg.word_embedding_dim = var_map["word_embedding_dim"].as<unsigned>();
    pg.enc_h_dim = var_map["enc_h_dim"].as<unsigned>();
    pg.enc_stacked_layer_num = var_map["enc_stacked_layer_num"].as<unsigned>();
    pg.dec_h_dim = var_map["dec_h_dim"].as<unsigned>();
    pg.dec_stacked_layer_num = var_map["dec_stacked_layer_num"].as<unsigned>();

    pg.enc_hidden_layer_output_dim = static_cast<unsigned>(pg.enc_h_dim * pg.enc_stacked_layer_num * 1.5);
    pg.enc_output_layer_output_dim = pg.dec_h_dim * pg.dec_stacked_layer_num; // output will be split to `dec_stacked_layer_num` pieces to init dec hidden   

}

template <typename RNNType>
void PoemGeneratorHandler<RNNType>::build_model()
{
    pg.EOS_idx = pg.word_dict.Convert(pg.EOS_STR) ;
    pg.build_model();
    pg.print_model_info();
}

template <typename RNNType>
void PoemGeneratorHandler<RNNType>::train(const std::vector<Poem> &poems , std::size_t max_epoch , std::size_t report_freq)
{
    std::size_t poems_size = poems.size();
    BOOST_LOG_TRIVIAL(info) << "train at " << poems_size << " poems" ;
    std::vector<std::size_t> access_order(poems_size);
    for (std::size_t idx = 0; idx < poems_size; ++idx) access_order[idx] = idx;
    cnn::MomentumSGDTrainer sgd(pg.m);
    std::size_t training_cnt = 0 ;
    for (std::size_t nr_epoch = 0; nr_epoch < max_epoch; ++nr_epoch)
    {
        BOOST_LOG_TRIVIAL(info) << "--------- " << nr_epoch + 1 << "/" << max_epoch << " ---------";
        shuffle(access_order.begin(), access_order.end(), rng);
        TimeStat stat;
        stat.start_time_stat();
        for (std::size_t idx = 0; idx < poems_size; ++idx)
        {
            std::size_t access_idx = access_order.at(idx);
            const Poem &poem = poems.at(access_idx);
            cnn::ComputationGraph cg;
            pg.build_graph(cg, poem);
            stat.loss += as_scalar(cg.forward());
            cg.backward();
            sgd.update(1.f);
            ++training_cnt ;
            if(0 == training_cnt % report_freq) 
            {
                BOOST_LOG_TRIVIAL(trace) << training_cnt << "has been trained since last report. " ;
                training_cnt = 0 ; // avoid overflow
            }
        }
        sgd.update_epoch();
        stat.end_time_stat();
        BOOST_LOG_TRIVIAL(info) << "---------- " << nr_epoch + 1 << " epoch end --------\n"
            << "Time cost " << stat.get_time_cost_in_seconds() << " s\n"
            << "sum E = " << stat.get_sum_E();
    }
    BOOST_LOG_TRIVIAL(info) << "training done ." ;
}

template <typename RNNType>
void PoemGeneratorHandler<RNNType>::generate(const std::string &first_seq, std::vector<std::string> &generated_poem)
{
    cnn::ComputationGraph cg;
    IndexSeq first_index_seq;
    Poem poem;

    // trans first_seq to indexSeq
    std::vector<std::string> words_cont;
    slice_utf8_sents2single_words(first_seq, words_cont);
    for (std::string &word : words_cont)
    {
        first_index_seq.push_back(pg.word_dict.Convert(word));
    }
    pg.generate(cg , first_index_seq, poem);
    // trans Poem to std::vector of sents 
    std::vector<std::string> tmp_generated_poem;
    for (IndexSeq &index_seq : poem)
    {
        std::string tmp_line = "";
        for (Index word_lookup_idx : index_seq)
        {
            tmp_line.append(pg.word_dict.Convert(word_lookup_idx));
        }
        tmp_generated_poem.push_back(tmp_line);
    }
    swap(tmp_generated_poem, generated_poem);
}

template <typename RNNType>
void PoemGeneratorHandler<RNNType>::save_model(std::ofstream &os)
{
    BOOST_LOG_TRIVIAL(info) << "saving model ..." ;
    boost::archive::text_oarchive to(os);
    to << pg.word_embedding_dim << pg.word_dict_size
        << pg.enc_h_dim << pg.enc_stacked_layer_num
        << pg.enc_hidden_layer_output_dim
        << pg.enc_output_layer_output_dim
        << pg.dec_h_dim << pg.dec_stacked_layer_num;
    to << pg.word_dict;
    to << (*pg.m);
    BOOST_LOG_TRIVIAL(info) << "saved ." ;
}

template <typename RNNType>
void PoemGeneratorHandler<RNNType>::load_model(std::ifstream &is)
{
    BOOST_LOG_TRIVIAL(info) << "loading model ..." ;
    boost::archive::text_iarchive ti(is);
    ti >> pg.word_embedding_dim >> pg.word_dict_size
        >> pg.enc_h_dim >> pg.enc_stacked_layer_num
        >> pg.enc_hidden_layer_output_dim
        >> pg.enc_output_layer_output_dim
        >> pg.dec_h_dim >> pg.dec_stacked_layer_num;
    ti >> pg.word_dict;
    assert(pg.word_dict.size() == pg.word_dict_size); 
    build_model();
    ti >> (*pg.m);
    BOOST_LOG_TRIVIAL(info) << "loaded ." ;
}

template <typename RNNType>
void PoemGeneratorHandler<RNNType>::slice_utf8_sents2single_words(const std::string &usent, 
        std::vector<std::string> &words_cont)
{
    std::vector<std::string> tmp_words_cont;
    std::string::const_iterator ite = usent.cbegin();
    while (true)
    {
        std::string::const_iterator pre_ite = ite;
        utf8::advance(ite, 1, usent.cend());
        std::string word(pre_ite, ite);
        if (word != " ") tmp_words_cont.push_back(word);
        if (ite == usent.cend()) break;
    }
    swap(words_cont, tmp_words_cont);
}
#endif
