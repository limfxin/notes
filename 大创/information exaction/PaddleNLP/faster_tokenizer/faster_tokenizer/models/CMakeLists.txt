cc_library(models
        SRCS wordpiece.cc faster_wordpiece.cc bpe.cc unigram.cc
        DEPS core json trie failure icuuc icudata lattice utils)
