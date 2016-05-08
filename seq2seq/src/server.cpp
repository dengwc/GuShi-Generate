#include <boost/program_options.hpp>
#include <boost/log/trivial.hpp>

#include "poem_generate.h"
#include "poem_generate_handler.h"

#include "thirdparty/mongoose.h"

using namespace std ;
namespace po = boost::program_options ;

// Mongoose
static char s_http_port[10] = "6669" ;
static void send_error_result(struct mg_connection *nc, const char *msg) ;
static void rest_api(struct mg_connection *nc , struct http_message *hm) ;
static void ev_handler(struct mg_connection *nc , int ev , void *ev_data) ;

// Poem Generator
using ModelHandler = PoemGeneratorHandler<cnn::SimpleRNNBuilder> ;
static shared_ptr<ModelHandler> p_pgh ;

static const string ProgramDescription = "Poem Generator Server ." ;

int main(int argc , char *argv[])
{
    string usage = ProgramDescription + "\n" +
        "usage : " + argv[0] + " <options>\n" +
        "options" ;
    po::options_description optparser(usage);
    optparser.add_options()
        ("port,p" , po::value<string>()->default_value("6669") , "specify port" )
        ("model,m" , po::value<string>(),"poem generator model path")
        ("cnn-mem" , po::value<unsigned>()->default_value(512U) , "specify cnn pre-allocator memory size")
        ("help,h" , "show help information") ;
    po::variables_map var_map ;
    po::store( po::command_line_parser(argc , argv).options(optparser).allow_unregistered().run() , var_map  ) ;
    po::notify(var_map) ;
    
    if(0 != var_map.count("help"))
    {
        cerr << optparser << endl ;
        return 0 ;
    }
    string model_path ;
    if(0 == var_map.count("port"))
    {
        cerr << "port should be specified" << endl ;
        cerr << optparser << endl ;
        return 1 ;
    }
    strcpy(s_http_port , var_map["port"].as<string>().c_str()) ;
    if(0 == var_map.count("model"))
    {
        cerr << "model should be specified" << endl ;
        cerr << optparser << endl ;
        return 1 ;
    }
    model_path = var_map["model"].as<string>() ;
    
    // load model 
    ifstream model_is(model_path) ;
    if(!model_is)
    {
        cerr << "failed to open model at path : `" << model_path << "` \n" ;
        return 1 ; 
    }
        // build argv for CNN
    int cnn_argc = 3 ;
    char *cnn_arg0 = argv[0] ;
    char cnn_arg1[] = "--cnn-mem" ;
    char cnn_arg2[10] ;
    strcpy(cnn_arg2 , to_string(var_map["cnn-mem"].as<unsigned>()).c_str()) ;
    char **cnn_argv = new char *[cnn_argc+1]{cnn_arg0 , cnn_arg1 , cnn_arg2 , NULL } ;
    cnn::Initialize( cnn_argc , cnn_argv , 1234);
    delete [] cnn_argv ;
    p_pgh.reset(new ModelHandler()) ;
    p_pgh->load_model(model_is) ; model_is.close() ;
    
    // build Server using Mongoose
    struct mg_mgr mgr ;
    shared_ptr<struct mg_mgr> p_mgr(&mgr , mg_mgr_free) ; // using shared_ptr to avoid realse mgr handly at every exit point
    struct mg_connection *nc ;

    mg_mgr_init(&mgr , NULL) ;
    nc = mg_bind(&mgr , s_http_port , ev_handler) ;
    if(nc == NULL)
    {
        cerr << "failed to listen at port "  << s_http_port <<  "\n"
             <<"port may need to be change . \n";
        return 1 ;
    }
    mg_set_protocol_http_websocket(nc) ;
    cerr << "starting RESTFful server on port " <<  s_http_port << endl  ;
    for(;;)
    {
        mg_mgr_poll(&mgr , 1000) ;
    }
    return 0 ;
}

static void send_error_result(struct mg_connection *nc, const char *msg) 
{
    mg_printf_http_chunk(nc, "Error: %s\n", msg);
    mg_send_http_chunk(nc, "", 0); /* Send empty chunk, the end of response */
}

static void rest_api(struct mg_connection *nc , struct http_message *hm)
{
    char first_seq[256] ;
    mg_get_http_var(&hm->body , "first_seq" , first_seq , sizeof(first_seq)) ;
    mg_printf(nc, "%s", "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n");
    if(first_seq[0] == '\0')
    {
        send_error_result(nc , "bad request") ;
    }
    else
    {
        //mg_printf_http_chunk(nc , "request value : %s\n" , first_seq) ;
        vector<string> poem ;
        p_pgh->generate(first_seq , poem) ;
        for(string &sent : poem)
        {
            mg_printf_http_chunk(nc , "%s\n" , sent.c_str()) ;
        }
    }
    mg_send_http_chunk(nc , "" , 0) ; // end chunked
}

static void ev_handler(struct mg_connection *nc , int ev , void *ev_data)
{
    struct http_message *hm = (struct http_message *)ev_data ;
    switch(ev)
    {
        case MG_EV_HTTP_REQUEST :
            if(0 == mg_vcmp(&hm->uri , "/"))
            {
                rest_api(nc , hm) ;
            }
            else
            {
                send_error_result(nc , "bad url") ;
            }
            break ;
        default :
            break ;
    }
}
