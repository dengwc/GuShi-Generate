INCLUDE_DIRECTORIES(${source_directory})

ADD_EXECUTABLE(poem_generate main.cpp poem_generate.cpp poem_generate_handler.cpp layers.cpp)

target_link_libraries(poem_generate cnn ${Boost_LIBRARIES})
