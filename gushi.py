# -*- coding:utf-8 -*-
import os
import sys
import firstSen as fs

def gushi_generate(topic):
    first_seq = fs.generate_first_sentence(topic)
    gushi = other_sentences(first_seq)
    print gushi

def other_sentences(first_seq):
    #cmd = "curl -d first_seq='%s' 0.0.0.0:6668" % unicode(first_seq,'utf-8')
    cmd = "curl -d first_seq='%s' 0.0.0.0:6668" % first_seq
    gushi = os.popen(cmd)
    return gushi.read()

if __name__=="__main__":
    gushi_generate(sys.argv[1])
