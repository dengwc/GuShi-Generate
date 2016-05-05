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


#include "layers.h"
#include "typedec.h"



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
    BILSTMLayer *bi_enc;
    cnn::LSTMBuilder *dec;
    DenseLayer *enc_hidden_layer;
    MergeMax3Layer *enc_output_layer;
    DenseLayer *dec_output_layer;

    cnn::LookupParameters *words_lookup_param;
    cnn::Parameters *DEC_SOS_param;
    cnn::Parameters *DEC_EOS_param;
    Index EOS_idx; // in word dict index ; we may want to decode an EOS at every end poem sentence ;
   
    cnn::Dict word_dict;

    const static unsigned MaxHistoryLen ; //  = 3 
    const static unsigned PoemSentNum;

    PoemGenerator();
    ~PoemGenerator();

    void build_model();
    void print_model_info();

    Expression build_graph(cnn::ComputationGraph &cg , const Poem &poem);
    void generate(cnn::ComputationGraph &cg, const IndexSeq &first_seq, Poem &generated_poem);
};


#endif
