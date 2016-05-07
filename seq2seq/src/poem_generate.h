#ifndef POEM_GENERATE_H_INCLUDED
#define POEM_GENERATE_H_INCLUDED
#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <deque>
#include <boost/log/trivial.hpp>

#include "layers.h"
#include "typedec.h"

template <typename RNNType>
struct PoemGenerator
{
    // init from outer
    unsigned word_embedding_dim;
    unsigned enc_h_dim;
    unsigned enc_stacked_layer_num;
    unsigned dec_h_dim;
    unsigned dec_stacked_layer_num;
    
    // init from inner by calculation
    unsigned enc_hidden_layer_output_dim;
    unsigned enc_output_layer_output_dim;

    unsigned word_dict_size;

    cnn::Model *m;
    BIRNNLayer<RNNType> *bi_enc;
    RNNType *dec;
    DenseLayer *enc_hidden_layer;
    MergeMax3Layer *enc_output_layer;
    DenseLayer *dec_output_layer;

    cnn::LookupParameters *words_lookup_param;
    cnn::Parameters *DEC_SOS_param;
    Index EOS_idx; // in word dict index ; we may want to decode an EOS at every end poem sentence ;
   
    cnn::Dict word_dict;

    // static 
    const static std::size_t MaxHistoryLen ; //  = 3 
    const static std::size_t PoemSentNum;
    const static std::string EOS_STR ; 
    const static std::string UNK_STR ;

    PoemGenerator();
    ~PoemGenerator();

    void build_model();
    void print_model_info();

    cnn::expr::Expression build_graph(cnn::ComputationGraph &cg , const Poem &poem);
    void generate(cnn::ComputationGraph &cg, const IndexSeq &first_seq, Poem &generated_poem);
};

template <typename RNNType>
const std::size_t PoemGenerator<RNNType>::MaxHistoryLen = 3U;
template <typename RNNType>
const std::size_t PoemGenerator<RNNType>::PoemSentNum = 4U;
template <typename RNNType>
const std::string PoemGenerator<RNNType>::EOS_STR = "EOS_STR";
template <typename RNNType>
const std::string PoemGenerator<RNNType>::UNK_STR = "UNK_STR";

template <typename RNNType>
PoemGenerator<RNNType>::PoemGenerator() :
    m(nullptr),
    bi_enc(nullptr),
    dec(nullptr),
    enc_hidden_layer(nullptr),
    enc_output_layer(nullptr),
    dec_output_layer(nullptr)
{}

template <typename RNNType>
PoemGenerator<RNNType>::~PoemGenerator()
{
    if (m) delete m;
    if (bi_enc) delete bi_enc;
    if (dec) delete dec;
    if (enc_hidden_layer) delete enc_hidden_layer;
    if (enc_output_layer) delete enc_output_layer;
    if (dec_output_layer) delete dec_output_layer;
}


template <typename RNNType>
void PoemGenerator<RNNType>::build_model()
{
    assert(word_dict.is_frozen());
    m = new cnn::Model();
    bi_enc = new BIRNNLayer<RNNType>(m, enc_stacked_layer_num, word_embedding_dim, enc_h_dim);
    dec = new RNNType(dec_stacked_layer_num, word_embedding_dim, dec_h_dim, m);
    
    enc_hidden_layer = new DenseLayer(m, enc_h_dim * enc_stacked_layer_num * 2,
        enc_hidden_layer_output_dim);
    enc_output_layer = new MergeMax3Layer(m, enc_hidden_layer_output_dim, enc_hidden_layer_output_dim,
        enc_hidden_layer_output_dim, enc_output_layer_output_dim);
    dec_output_layer = new DenseLayer(m, dec_h_dim , word_dict_size);

    words_lookup_param = m->add_lookup_parameters(word_dict_size, { word_embedding_dim });
    DEC_SOS_param = m->add_parameters({ word_embedding_dim }); // SOS will be input , so param is needed
}

template <typename RNNType>
void PoemGenerator<RNNType>::print_model_info()
{
    BOOST_LOG_TRIVIAL(info) << "----------------- Model Info ---------------\n"
        << "word vocabulary size : " << word_dict_size << " with dimension : " << word_embedding_dim << "\n"
        << "encoder stacked layer number : " << enc_stacked_layer_num << " with dimension : " << enc_h_dim << "\n"
        << "encoder hidden layer output dim : " << enc_hidden_layer_output_dim << "\n"
        << "encoder output layer output dim : " << enc_output_layer_output_dim << "\n"
        << "decoder stacked layer number : " << dec_stacked_layer_num << " with dimensiom : " << dec_h_dim  ;
}

template <typename RNNType>
Expression PoemGenerator<RNNType>::build_graph(cnn::ComputationGraph &cg , const Poem &poem)
{
    bi_enc->new_graph(cg);
    dec->new_graph(cg);
    enc_hidden_layer->new_graph(cg);
    enc_output_layer->new_graph(cg);
    dec_output_layer->new_graph(cg);
    
    cnn::expr::Expression DEC_SOS_exp = parameter(cg, DEC_SOS_param);
    std::vector<cnn::expr::Expression> loss_cont;
    std::deque<cnn::expr::Expression> enc_hidden_layer_output_cont ;
    for (std::size_t generating_idx = 1; generating_idx < poem.size(); ++generating_idx)
    {
        const IndexSeq &cur_seq = poem.at(generating_idx-1),
            &gen_seq = poem.at(generating_idx);
        std::vector<cnn::expr::Expression> X(cur_seq.size());
        for (size_t word_idx = 0; word_idx < cur_seq.size(); ++word_idx)
        {
            X[word_idx] = lookup(cg, words_lookup_param, cur_seq.at(word_idx));
        }

        // BILSTM encode layer
        bi_enc->start_new_sequence();
        bi_enc->build_graph(X);
        
        // enc hidden layer
        std::vector<cnn::expr::Expression> final_h_cont;
        bi_enc->get_final_h(final_h_cont); // 2 * enc_stacked_layer_num , every is {enc_h_dim , 1}
        cnn::expr::Expression h_combined_exp = concatenate(final_h_cont); // {enc_h_dim * 2 * enc_stacked_layer_num , 1}
        cnn::expr::Expression enc_hidden_layer_output_exp = enc_hidden_layer->build_graph(h_combined_exp);
        cg.incremental_forward(); // do as example 
        enc_hidden_layer_output_exp = rectify(enc_hidden_layer_output_exp); // rectify

        enc_hidden_layer_output_cont.push_front(enc_hidden_layer_output_exp);
        
        // enc output layer
        std::size_t cur_history_size = enc_hidden_layer_output_cont.size() ;
        enc_hidden_layer_output_cont.resize( std::min( MaxHistoryLen, cur_history_size) ); // truncate the first MaxHistoyrLen
        cnn::expr::Expression enc_output_layer_output_exp = enc_output_layer->build_graph(std::vector<Expression>(
            enc_hidden_layer_output_cont.begin(), enc_hidden_layer_output_cont.end()));

        // split output exp to init the decoder
        std::vector<cnn::expr::Expression> init_for_dec_combine(dec_stacked_layer_num);
        for (std::size_t layer_idx = 0; layer_idx < dec_stacked_layer_num; ++layer_idx)
        {
            cnn::expr::Expression splited_exp = pickrange(enc_output_layer_output_exp,
                layer_idx * dec_h_dim, (layer_idx + 1)*dec_h_dim);
            init_for_dec_combine[layer_idx] = cnn::expr::tanh(splited_exp);
        }
        dec->start_new_sequence(init_for_dec_combine);

        // output
        cnn::expr::Expression pre_word_exp = DEC_SOS_exp;
        for (size_t word_idx = 0; word_idx < gen_seq.size(); ++word_idx)
        {
            // to generate `word_idx` word
            cnn::expr::Expression dec_out_exp = dec->add_input(pre_word_exp);
            cnn::expr::Expression dec_output_layer_output_exp = dec_output_layer->build_graph(dec_out_exp);
            cnn::expr::Expression loss = pickneglogsoftmax(dec_output_layer_output_exp , gen_seq.at(word_idx));
            loss_cont.push_back(loss);
            pre_word_exp = lookup(cg, words_lookup_param, gen_seq.at(word_idx)); // for next circulation
        }
            // processing EOS
            // here pre_word_exp is the last word exp
        /* MAY BE LEAD to EOS prediction 
        cnn::expr::Expression dec_out_exp = dec->add_input(pre_word_exp);
        cnn::expr::Expression dec_output_layer_output_exp = dec_output_layer->build_graph(dec_out_exp);
        cnn::expr::Expression loss = pickneglogsoftmax(dec_output_layer_output_exp, EOS_idx);
        loss_cont.push_back(loss);
        */
    }
    return cnn::expr::sum(loss_cont);
}

template <typename RNNType>
void PoemGenerator<RNNType>::generate(cnn::ComputationGraph &cg, const IndexSeq &first_seq, Poem &generated_poem)
{
    size_t poem_sent_len = first_seq.size();
    
    bi_enc->new_graph(cg);
    enc_hidden_layer->new_graph(cg);
    enc_output_layer->new_graph(cg);
    dec->new_graph(cg);
    dec_output_layer->new_graph(cg);

    cnn::expr::Expression DEC_SOS_exp = parameter(cg, DEC_SOS_param);
    std::deque<cnn::expr::Expression> history_outputs;
    std::vector<IndexSeq> tmp_poem(PoemSentNum, IndexSeq(poem_sent_len));
    std::copy(first_seq.cbegin(), first_seq.cend(), tmp_poem[0].begin());
    for (unsigned generating_idx = 1; generating_idx < PoemSentNum; ++generating_idx)
    {
        IndexSeq &cur_seq = tmp_poem.at(generating_idx - 1),
            &gen_seq = tmp_poem.at(generating_idx);
        // ready input for encoder
        std::vector<cnn::expr::Expression> X(poem_sent_len);
        for (std::size_t word_idx = 0; word_idx < poem_sent_len; ++word_idx)
        {
            X[word_idx] = lookup(cg, words_lookup_param, cur_seq.at(word_idx));
        }

        // bilstm encoder
        bi_enc->start_new_sequence();
        bi_enc->build_graph(X);

        // encoder hidden layer
        std::vector<Expression> final_h_cont;
        bi_enc->get_final_h(final_h_cont);
        cnn::expr::Expression h_combined = concatenate(final_h_cont);
        cnn::expr::Expression enc_hidden_layer_output = enc_hidden_layer->build_graph(h_combined);
        enc_hidden_layer_output = rectify(enc_hidden_layer_output);
        
        // encoder output layer
        history_outputs.push_front(enc_hidden_layer_output);
        size_t cur_history_size = history_outputs.size() ;
        history_outputs.resize(std::min(MaxHistoryLen, cur_history_size));
        cnn::expr::Expression enc_output_layer_output = enc_output_layer->build_graph(std::vector<Expression>(history_outputs.begin(), history_outputs.end()));

        // decoder
        //std::vector<Expression> splited_exp_cont(dec_stacked_layer_num * 2); For RNN or GRU , only h is to be initilized
        std::vector<Expression> splited_exp_cont(dec_stacked_layer_num );
        for (size_t layer_idx = 0; layer_idx < dec_stacked_layer_num; ++layer_idx)
        {
            cnn::expr::Expression init_for_c = pickrange(enc_output_layer_output, layer_idx * dec_h_dim,
                (layer_idx + 1) * dec_h_dim);
            splited_exp_cont[layer_idx] = cnn::expr::tanh(init_for_c);
        }
        dec->start_new_sequence(splited_exp_cont);
        cg.incremental_forward();
        cnn::expr::Expression pre_word_exp = DEC_SOS_exp;
        for (size_t gen_idx = 0; gen_idx < poem_sent_len; ++gen_idx)
        {
            cnn::expr::Expression dec_out_exp = dec->add_input(pre_word_exp);
            dec_output_layer->build_graph(dec_out_exp); 
            std::vector<cnn::real> dist = as_vector(cg.incremental_forward());
            Index predicted_word_idx = distance(dist.cbegin(), max_element(dist.cbegin(), dist.cend()));
            gen_seq[gen_idx] = predicted_word_idx;
            pre_word_exp = lookup(cg, words_lookup_param, predicted_word_idx);
        }
    }
    swap(tmp_poem, generated_poem);
}

#endif
