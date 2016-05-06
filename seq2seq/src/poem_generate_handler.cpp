#include <boost/log/trivial.hpp>
#include <boost/log/core.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "poem_generate_handler.h"
#include "timestat.hpp"
#include "thirdparty/utf8.h"

using namespace std;
using namespace cnn;

PoemGeneratorHandler::PoemGeneratorHandler(PoemGenerator &pg , size_t seed)
    :pg(pg) , rng(seed)
{}

PoemGeneratorHandler::~PoemGeneratorHandler()
{}

void PoemGeneratorHandler::read_train_data(ifstream &is, vector<Poem> &poems)
{
    vector<Poem> tmp_poems;
    tmp_poems.reserve(0x20000); // 2^17 
    string line;
    while (getline(is, line))
    {
        vector<string> sent_cont;
        boost::split(sent_cont, line, boost::is_any_of("\t"));
        Poem poem(sent_cont.size());
        for (size_t sent_idx = 0 ; sent_idx < sent_cont.size() ; ++sent_idx)
        {
            const string &sent = sent_cont.at(sent_idx);
            vector<string> words_cont;
            boost::split(words_cont, sent, boost::is_any_of(" "));
            IndexSeq &cur_sent = poem.at(sent_idx);
            cur_sent.resize(words_cont.size());
            for (size_t word_idx = 0; word_idx < words_cont.size(); ++word_idx)
            {
                cur_sent.at(word_idx) = pg.word_dict.Convert(words_cont.at(word_idx)); // add to dict and buid poems
            }
        }
        tmp_poems.push_back(poem);
    }
    swap(tmp_poems, poems);
}

void PoemGeneratorHandler::finish_reading_training_data()
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

void PoemGeneratorHandler::finish_reading_training_data(boost::program_options::variables_map &var_map)
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

void PoemGeneratorHandler::build_model()
{
    pg.EOS_idx = pg.word_dict.Convert(pg.EOS_STR) ;
    pg.build_model();
    pg.print_model_info();
}

void PoemGeneratorHandler::train(const std::vector<Poem> &poems , size_t max_epoch , size_t report_freq)
{
    size_t poems_size = poems.size();
    BOOST_LOG_TRIVIAL(info) << "train at " << poems_size << " poems" ;
    vector<size_t> access_order(poems_size);
    for (size_t idx = 0; idx < poems_size; ++idx) access_order[idx] = idx;
    SimpleSGDTrainer sgd(pg.m);
    size_t training_cnt = 0 ;
    for (size_t nr_epoch = 0; nr_epoch < max_epoch; ++nr_epoch)
    {
        BOOST_LOG_TRIVIAL(info) << "--------- " << nr_epoch + 1 << "/" << max_epoch << " ---------";
        shuffle(access_order.begin(), access_order.end(), rng);
        TimeStat stat;
        stat.start_time_stat();
        for (size_t idx = 0; idx < poems_size; ++idx)
        {
            size_t access_idx = access_order.at(idx);
            const Poem &poem = poems.at(access_idx);
            ComputationGraph cg;
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

void PoemGeneratorHandler::generate(const string &first_seq, vector<string> &generated_poem)
{
    ComputationGraph cg;
    IndexSeq first_index_seq;
    Poem poem;

    // trans first_seq to indexSeq
    vector<string> words_cont;
    slice_utf8_sents2single_words(first_seq, words_cont);
    for (string &word : words_cont)
    {
        first_index_seq.push_back(pg.word_dict.Convert(word));
    }
    pg.generate(cg , first_index_seq, poem);
    // trans Poem to vector of sents 
    vector<string> tmp_generated_poem;
    for (IndexSeq &index_seq : poem)
    {
        string tmp_line = "";
        for (Index word_lookup_idx : index_seq)
        {
            tmp_line.append(pg.word_dict.Convert(word_lookup_idx));
        }
        tmp_generated_poem.push_back(tmp_line);
    }
    swap(tmp_generated_poem, generated_poem);
}


void PoemGeneratorHandler::save_model(std::ofstream &os)
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

void PoemGeneratorHandler::load_model(std::ifstream &is)
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

void PoemGeneratorHandler::slice_utf8_sents2single_words(const std::string &usent, std::vector<string> &words_cont)
{
    vector<string> tmp_words_cont;
    string::const_iterator ite = usent.cbegin();
    while (true)
    {
        string::const_iterator pre_ite = ite;
        utf8::advance(ite, 1, usent.cend());
        string word(pre_ite, ite);
        if (word != " ") tmp_words_cont.push_back(word);
        if (ite == usent.cend()) break;
    }
    swap(words_cont, tmp_words_cont);
}
