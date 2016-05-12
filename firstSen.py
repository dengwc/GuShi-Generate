# -*- coding:utf-8 -*-
# Generate first sentence of gushi, given topic
import os
import sys
import random

SENSIZE = 5
LISTSIZE = 20
TOPICFILE = './data/Shixuehanying.txt.jian'
SRILMDIR = '../srilm/bin/i686-m64/ngram'
LMDIR = './lm/gushi.lm'

def generate_first_sentence(topic):
    keywords_list = topic_load(topic)
    candidate_sentence = generate_sentence(keywords_list)
    best_sentence = find_best_sentence(candidate_sentence)
    return best_sentence.replace(' ','')

def topic_load(topic):
    ### load topic and keywords
    fp = open(TOPICFILE)

    find_flag = False
    for line in fp.readlines():
        if 'begin' in line and topic in line:
            keywords_list = []
            find_flag = True
        elif 'end' in line and find_flag==True:
            return keywords_list
        elif find_flag==True:
            keywordStr = line.strip('\r\n').split('\t')[-1]
            keywords_list += keywordStr.split(' ')

    print 'Not Found topic:',topic
    return topic_load('丽人类')

def generate_sentence(keywords_list):
    ### generate candidate sentence by combination
    candidate_sentence = []
    two_orign_list = []
    three_orign_list = []

    # split 2 word and 3 word
    for keyword in keywords_list:
        if len(keyword)==6:
            two_orign_list.append(keyword)
        elif len(keyword)==9:
            three_orign_list.append(keyword)

    two_size = min(len(two_orign_list),LISTSIZE)
    three_size = min(len(three_orign_list),LISTSIZE)
    
    # max size
    two_list = random.sample(two_orign_list,two_size)
    three_list = random.sample(three_orign_list,three_size)

    # 2 + 3 / 3 + 2
    for two in two_list:
        for three in three_list:
            candidate_sentence.append(two+three)
            candidate_sentence.append(three+two)

    two_list = random.sample(two_orign_list,two_size)
    three_list = random.sample(three_orign_list,three_size)

    # 3+3
    for wordi in three_list:
        for wordj in three_list:
            if wordi==wordj:
                continue
            candidate_sentence.append(six_char_2_five_char(wordi+wordj))
            candidate_sentence.append(six_char_2_five_char(wordj+wordi))

    two_list = random.sample(two_orign_list,two_size)
    three_list = random.sample(three_orign_list,three_size)

    # 2+2+2
    for wordi in two_list:
        for wordj in two_list:
            for wordk in two_list:
                if wordi==wordj or wordi== wordk or wordj==wordk:
                    continue
                candidate_sentence.append(six_char_2_five_char(wordi+wordj+wordk))
                candidate_sentence.append(six_char_2_five_char(wordi+wordk+wordj))
                candidate_sentence.append(six_char_2_five_char(wordj+wordi+wordk))
                candidate_sentence.append(six_char_2_five_char(wordj+wordk+wordi))
                candidate_sentence.append(six_char_2_five_char(wordk+wordj+wordi))
                candidate_sentence.append(six_char_2_five_char(wordk+wordi+wordj))

    return candidate_sentence

def find_best_sentence(candidate_sentence):
    ### find best sentence by n-gram

    # create candidate sentence file
    if 'tmp' not in os.listdir('./'):
        os.mkdir('./tmp')
    fw = open('./tmp/gushi.txt','w+')
    for sen in candidate_sentence:
        fw.write(segment(sen)+'\n')
    fw.close()

    # n-gram score
    cmd = '%s -ppl ./tmp/gushi.txt -debug 1 -order 4 -lm %s > ./tmp/gushi.ppl' % (SRILMDIR,LMDIR)
    os.system(cmd)

    # find best score
    fp = open('./tmp/gushi.ppl')
    score_list = []
    gushi_list = []
    for line in fp.readlines():
        if 'file' in line:
            break
        if 'words' in line or line=='\n':
            continue
        elif 'ppl' in line:
            score_list.append(float(line.strip('\r\n').split(' ')[-1]))
        else:
            gushi_list.append(line.strip())

    #print len(score_list)
    #print len(gushi_list)

    index = score_list.index(min(score_list))
    return gushi_list[index]

def six_char_2_five_char(six):
    return six[:3*SENSIZE]

def segment(line):
    ### segment for line
    parts = []
    for i in range(len(line)/3):
        parts.append(line[i*3:i*3+3])
    return ' '.join(parts)

if __name__=='__main__':
    #generate_first_sentence('丽人类')
    generate_first_sentence(sys.argv[1])
