#include "poem_generate.h"
#include "poem_generate_handler.h"
#include <boost/program_options.hpp>
#include <boost/log/trivial.hpp>
using namespace std;
namespace po = boost::program_options;
const string PROGRAM_DESCRIPTION = "Chinese Poem Generator based on CNN Library";

int train_process(int argc, char *argv[], const string &program_name)
{
    string description = PROGRAM_DESCRIPTION + "\n"
        "Training process .\n"
        "using `" + program_name + " train <options>` to train . Training options are as following";
    po::options_description op_des = po::options_description(description);
    op_des.add_options()
        ("training_data", po::value<string>(), "The path to training data")
        ("max_epoch", po::value<unsigned>()->default_value(4), "The epoch to iterate for training")
        ("model", po::value<string>(), "Use to specify the model name(path)")
        ("word_embedding_dim", po::value<unsigned>()->default_value(50), "The dimension for dynamic channel word embedding.")
        ("enc_stacked_layer_num", po::value<unsigned>()->default_value(3), "The number of stacked layers in encoder bi-LSTM.")
        ("enc_h_dim", po::value<unsigned>()->default_value(500), "The dimension for encoder bi-LSTM H.")
        ("dec_stacked_layer_num", po::value<unsigned>()->default_value(3), "The number of stacked layers in decoder LSTM.")
        ("dec_h_dim", po::value<unsigned>()->default_value(500), "The dimension for decoder LSTM H.")
        ("help,h", "Show help information.");
    po::variables_map var_map;
    po::store(po::command_line_parser(argc, argv).options(op_des).allow_unregistered().run(), var_map);
    po::notify(var_map);
    if (var_map.count("help"))
    {
        cerr << op_des << endl;
        return 0;
    }
    // set params 
    string training_data_path;
    if (0 == var_map.count("training_data"))
    {
        BOOST_LOG_TRIVIAL(fatal) << "Error : Training data should be specified ! \n"
            "using `" + program_name + " train -h ` to see detail parameters .\n"
            "Exit .";
        return -1;
    }
    training_data_path = var_map["training_data"].as<string>();

    unsigned max_epoch = var_map["max_epoch"].as<unsigned>();

    // Init 
    cnn::Initialize(argc, argv, 1234); // 
    PoemGenerator pg;
    PoemGeneratorHandler pgh(pg);

    ifstream train_is(training_data_path);
    if (!train_is) {
        BOOST_LOG_TRIVIAL(fatal) << "failed to open training: `" << training_data_path << "` .\n Exit! \n";
        return -1;
    }
    vector<Poem> poems;
    pgh.read_train_data(train_is , poems);
    train_is.close();
    // set model structure param 
    pgh.finish_reading_training_data(var_map);

    // build model structure
    pgh.build_model(); // passing the var_map to specify the model structure


                                 // reading developing data
    // Train 
    pgh.train(poems , max_epoch);

    // save model
    string model_path;
    if (0 == var_map.count("model"))
    {
        cerr << "no model name specified . using default .\n";
        ostringstream oss;
        oss << "poemgen_" << pg.word_embedding_dim << "_" << pg.enc_h_dim
            << "_" << pg.dec_h_dim << ".model";
        model_path = oss.str();
    }
    else model_path = var_map["model"].as<string>();
    ofstream os(model_path);
    if (!os)
    {
        BOOST_LOG_TRIVIAL(fatal) << "failed to open model path at '" << model_path << "'. \n Exit !";
        return -1;
    }
    pgh.save_model(os);
    os.close();
    return 0;
}

int generate_process(int argc, char *argv[], const string &program_name)
{
    string description = PROGRAM_DESCRIPTION + "\n"
        "Generate process ."
        "using `" + program_name + " generate <options>` to generate poems . generate options are as following";
    po::options_description op_des = po::options_description(description);
    op_des.add_options()
        ("first_seq", po::value<string>(), "The first sequence .")
        ("model", po::value<string>(), "Use to specify the model name(path)")
        ("help,h", "Show help information.");
    po::variables_map var_map;
    po::store(po::command_line_parser(argc, argv).options(op_des).allow_unregistered().run(), var_map);
    po::notify(var_map);
    if (var_map.count("help"))
    {
        cerr << op_des << endl;
        return 0;
    }

    //set params 
    string first_seq , model_path;
    if (0 == var_map.count("first_seq"))
    {
        BOOST_LOG_TRIVIAL(fatal) << "raw_data path should be specified .\n"
            "Exit!";
        return -1;
    }
    else first_seq = var_map["first_seq"].as<string>();

    if (0 == var_map.count("model"))
    {
        BOOST_LOG_TRIVIAL(fatal) << "Model path should be specified ! \n"
            "Exit! ";
        return -1;
    }
    else model_path = var_map["model"].as<string>();

    // Init 
    cnn::Initialize(argc, argv, 1234);
    PoemGenerator pg;
    PoemGeneratorHandler pgh(pg);



    // load model 
    ifstream is(model_path);
    if (!is)
    {
        BOOST_LOG_TRIVIAL(fatal) << "Failed to open model path at '" << model_path << "' . \n"
            "Exit .";
        return -1;
    }
    pgh.load_model(is);
    is.close();
    vector<string> generated_poem;
    pgh.generate(first_seq, generated_poem);
    for (size_t idx = 0; idx < generated_poem.size(); ++idx)
    {
        cout << generated_poem.at(idx) << endl;
    }
    return 0;
}

int main(int argc, char *argv[])
{
    string usage = PROGRAM_DESCRIPTION + "\n"
        "usage : " + string(argv[0]) + " [ train | generate ] <options> \n"
        "using  `" + string(argv[0]) + " [ train | generate ] -h` to see details for specify task\n";
    if (argc <= 1)
    {
        cerr << usage;
        return -1;
    }
    else if (string(argv[1]) == "train") return train_process(argc - 1, argv + 1, argv[0]);
    else if (string(argv[1]) == "generate") return generate_process(argc - 1, argv + 1, argv[0]);
    else
    {
        cerr << "unknown mode : " << argv[1] << "\n"
            << usage;
        return -1;
    }
}