#include <deque>
#include "poem_generate.h"
#include <boost/log/trivial.hpp>
using namespace std;
using namespace cnn;

const unsigned PoemGenerator::MaxHistoryLen = 3U;
const unsigned PoemGenerator::PoemSentNum = 4U;

PoemGenerator::PoemGenerator() :
    m(nullptr),
    bi_enc(nullptr),
    dec(nullptr),
    enc_hidden_layer(nullptr),
    enc_output_layer(nullptr),
    dec_output_layer(nullptr)
{}

PoemGenerator::~PoemGenerator()
{
    if (m) delete m;
    if (bi_enc) delete bi_enc;
    if (dec) delete dec;
    if (enc_hidden_layer) delete enc_hidden_layer;
    if (enc_output_layer) delete enc_output_layer;
    if (dec_output_layer) delete dec_output_layer;
}



void PoemGenerator::build_model()
{
    assert(word_dict.is_frozen());
    m = new Model();
    bi_enc = new BILSTMLayer(m, enc_stacked_layer_num, word_embedding_dim, enc_h_dim);
    dec = new LSTMBuilder(dec_stacked_layer_num, word_embedding_dim, dec_h_dim, m);
    
    enc_hidden_layer = new DenseLayer(m, enc_h_dim * enc_stacked_layer_num * 2,
        enc_hidden_layer_output_dim);
    enc_output_layer = new MergeMax3Layer(m, enc_hidden_layer_output_dim, enc_hidden_layer_output_dim,
        enc_hidden_layer_output_dim, enc_output_layer_output_dim);
    dec_output_layer = new DenseLayer(m, dec_h_dim , word_dict_size);

    words_lookup_param = m->add_lookup_parameters(word_dict_size, { word_embedding_dim });
    DEC_SOS_param = m->add_parameters({ word_embedding_dim });
    DEC_EOS_param = m->add_parameters({ word_embedding_dim });
}

void PoemGenerator::print_model_info()
{
    BOOST_LOG_TRIVIAL(info) << "----------------- Model Info ---------------\n"
        << "word vocabulary size : " << word_dict_size << " with dimension : " << word_embedding_dim << "\n"
        << "encoder stacked layer number : " << enc_stacked_layer_num << " with dimension : " << enc_h_dim << "\n"
        << "encoder hidden layer output dim : " << enc_hidden_layer_output_dim << "\n"
        << "encoder output layer output dim : " << enc_output_layer_output_dim << "\n"
        << "decoder stacked layer number : " << dec_stacked_layer_num << " with dimensiom : " << dec_h_dim  ;
}

Expression PoemGenerator::build_graph(ComputationGraph &cg , const Poem &poem)
{
    bi_enc->new_graph(cg);
    dec->new_graph(cg);
    enc_hidden_layer->new_graph(cg);
    enc_output_layer->new_graph(cg);
    dec_output_layer->new_graph(cg);
    
    Expression DEC_SOS_exp = parameter(cg, DEC_SOS_param);
    Expression DEC_EOS_exp = parameter(cg, DEC_EOS_param);
    vector<Expression> loss_cont;
    deque<Expression> enc_hidden_layer_output_cont ;
    for (size_t generating_idx = 1; generating_idx < poem.size(); ++generating_idx)
    {
        const IndexSeq &cur_seq = poem.at(generating_idx-1),
            &gen_seq = poem.at(generating_idx);
        vector<Expression> X(cur_seq.size());
        for (size_t word_idx = 0; word_idx < cur_seq.size(); ++word_idx)
        {
            X[word_idx] = lookup(cg, words_lookup_param, cur_seq.at(word_idx));
        }

        // BILSTM encode layer
        bi_enc->start_new_sequence();
        bi_enc->build_graph(X);
        
        // enc hidden layer
        vector<Expression> final_h_cont;
        bi_enc->get_final_h(final_h_cont); // 2 * enc_stacked_layer_num , every is {enc_h_dim , 1}
        Expression h_combined_exp = concatenate(final_h_cont); // {enc_h_dim * 2 * enc_stacked_layer_num , 1}
        Expression enc_hidden_layer_output_exp = enc_hidden_layer->build_graph(h_combined_exp);
        cg.incremental_forward(); // do as example 
        enc_hidden_layer_output_exp = rectify(enc_hidden_layer_output_exp); // rectify

        enc_hidden_layer_output_cont.push_front(enc_hidden_layer_output_exp);
        
        // enc output layer
        enc_hidden_layer_output_cont.resize(min(MaxHistoryLen, enc_hidden_layer_output_cont.size())); // truncate the first MaxHistoyrLen
        Expression enc_output_layer_output_exp = enc_output_layer->build_graph(vector<Expression>(
            enc_hidden_layer_output_cont.begin(), enc_hidden_layer_output_cont.end()));

        // split output exp to init the decoder
        vector<Expression> init_for_dec_combine(dec_stacked_layer_num * 2);
        for (size_t layer_idx = 0; layer_idx < dec_stacked_layer_num; ++layer_idx)
        {
            Expression splited_exp = pickrange(enc_output_layer_output_exp,
                layer_idx * dec_h_dim, (layer_idx + 1)*dec_h_dim);
            init_for_dec_combine[layer_idx] = splited_exp;
            init_for_dec_combine[layer_idx + dec_stacked_layer_num] = tanh(splited_exp);
        }
        dec->start_new_sequence(init_for_dec_combine);

        // output
        Expression pre_word_exp = DEC_SOS_exp;
        for (size_t word_idx = 0; word_idx < gen_seq.size(); ++word_idx)
        {
            // to generate `word_idx` word
            Expression dec_out_exp = dec->add_input(pre_word_exp);
            Expression dec_output_layer_output_exp = dec_output_layer->build_graph(dec_out_exp);
            Expression loss = pickneglogsoftmax(dec_output_layer_output_exp , gen_seq.at(word_idx));
            loss_cont.push_back(loss);
            pre_word_exp = lookup(cg, words_lookup_param, gen_seq.at(word_idx)); // for next circulation
        }
            // processing EOS
            // here pre_word_exp is the last word exp
        Expression dec_out_exp = dec->add_input(pre_word_exp);
        Expression dec_output_layer_output_exp = dec_output_layer->build_graph(dec_out_exp);
        Expression loss = pickneglogsoftmax(dec_output_layer_output_exp, EOS_idx);
        loss_cont.push_back(loss);
    }
    return sum(loss_cont);
}

void PoemGenerator::generate(cnn::ComputationGraph &cg, const IndexSeq &first_seq, Poem &generated_poem)
{
    size_t poem_sent_len = first_seq.size();
    
    bi_enc->new_graph(cg);
    enc_hidden_layer->new_graph(cg);
    enc_output_layer->new_graph(cg);
    dec->new_graph(cg);
    dec_output_layer->new_graph(cg);

    Expression DEC_SOS_exp = parameter(cg, DEC_SOS_param);
    Expression DEC_EOS_exp = parameter(cg, DEC_EOS_param);
    deque<Expression> history_outputs;
    vector<IndexSeq> tmp_poem(PoemSentNum, IndexSeq(poem_sent_len));
    copy(first_seq.cbegin(), first_seq.cend(), tmp_poem[0].begin());
    for (unsigned generating_idx = 1; generating_idx < PoemSentNum; ++generating_idx)
    {
        IndexSeq &cur_seq = tmp_poem.at(generating_idx - 1),
            &gen_seq = tmp_poem.at(generating_idx);
        // ready input for encoder
        vector<Expression> X(poem_sent_len);
        for (size_t word_idx = 0; word_idx < poem_sent_len; ++word_idx)
        {
            X[word_idx] = lookup(cg, words_lookup_param, cur_seq.at(word_idx));
        }

        // bilstm encoder
        bi_enc->start_new_sequence();
        bi_enc->build_graph(X);

        // encoder hidden layer
        vector<Expression> final_h_cont;
        bi_enc->get_final_h(final_h_cont);
        Expression h_combined = concatenate(final_h_cont);
        Expression enc_hidden_layer_output = enc_hidden_layer->build_graph(h_combined);
        enc_hidden_layer_output = rectify(enc_hidden_layer_output);
        
        // encoder output layer
        history_outputs.push_front(enc_hidden_layer_output);
        history_outputs.resize(min(MaxHistoryLen, history_outputs.size()));
        Expression enc_output_layer_output = enc_output_layer->build_graph(vector<Expression>(history_outputs.begin(), history_outputs.end()));

        // decoder
        vector<Expression> splited_exp_cont(dec_stacked_layer_num * 2);
        for (size_t layer_idx = 0; layer_idx < dec_stacked_layer_num; ++layer_idx)
        {
            Expression init_for_c = pickrange(enc_output_layer_output, layer_idx * dec_h_dim,
                (layer_idx + 1) * dec_h_dim);
            splited_exp_cont[layer_idx] = init_for_c;
            splited_exp_cont[layer_idx + dec_stacked_layer_num] = tanh(init_for_c);
        }
        dec->start_new_sequence(splited_exp_cont);
        cg.incremental_forward();
        Expression pre_word_exp = DEC_SOS_exp;
        for (size_t gen_idx = 0; gen_idx < poem_sent_len; ++gen_idx)
        {
            Expression dec_out_exp = dec->add_input(pre_word_exp);
            Expression dec_output_layer_output_exp = dec_output_layer->build_graph(dec_out_exp);
            vector<cnn::real> dist = as_vector(cg.incremental_forward());
            Index predicted_word_idx = distance(dist.cbegin(), max_element(dist.cbegin(), dist.cend()));
            gen_seq[gen_idx] = predicted_word_idx;
            pre_word_exp = lookup(cg, words_lookup_param, predicted_word_idx);
        }
    }
    swap(tmp_poem, generated_poem);
}
