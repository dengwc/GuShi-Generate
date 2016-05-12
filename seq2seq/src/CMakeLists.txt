INCLUDE_DIRECTORIES(${source_directory})

ADD_EXECUTABLE(poem_generate main.cpp poem_generate.cpp poem_generate_handler.cpp layers.cpp)
ADD_EXECUTABLE(server server.cpp thirdparty/mongoose.c 
                                 poem_generate.cpp poem_generate_handler.cpp layers.cpp)

target_link_libraries(poem_generate cnn ${Boost_LIBRARIES})
target_link_libraries(server cnn ${Boost_LIBRARIES})

