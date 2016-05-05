INCLUDE_DIRECTORIES(${source_directory})

ADD_EXECUTABLE(poem_generate poem_generate.cpp layers.cpp)

target_link_libraries(poem_generate cnn ${Boost_LIBRARIES})