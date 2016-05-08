#include "thirdparty/mongoose.h"

static const char *s_http_port = "6669" ;

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
        mg_printf_http_chunk(nc , "request value : %s" , first_seq) ;
        mg_send_http_chunk(nc , "" , 0) ;
    }
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

int main(int argc , char *argv[])
{
    struct mg_mgr mgr ;
    struct mg_connection *nc ;

    mg_mgr_init(&mgr , NULL) ;
    nc = mg_bind(&mgr , s_http_port , ev_handler) ;
    if(nc == NULL)
    {
        fprintf( stderr ,"%s\n" ,"failed to listen . `port` may need to be change . ") ;
        mg_mgr_free(&mgr) ;
        exit(1) ;
    }
    mg_set_protocol_http_websocket(nc) ;
    
    for(int i = 1 ; i < argc ; ++i)
    {
        if(strcmp(argv[i] , "-p") == 0 && i+1 < argc )
        {
            s_http_port = argv[++i] ;
        }
    }
    
    printf("starting RESTFful server on port %s\n" , s_http_port) ;
    for(;;)
    {
        mg_mgr_poll(&mgr , 1000) ;
    }
    mg_mgr_free(&mgr) ;
    return 0 ;
}
