#!/usr/bin/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# LASER  Language-Agnostic SEntence Representations
# is a toolkit to calculate multilingual sentence embeddings
# and to use them for document classification, bitext filtering
# and mining
#
# --------------------------------------------------------
#
# Helper functions for tokenization and BPE

import os
import sys
import tempfile
import fastBPE
import numpy as np
from subprocess import run, check_output, DEVNULL

# get environment
assert os.environ.get('LASER'), 'Please set the enviornment variable LASER'
LASER = os.environ['LASER']

FASTBPE = f'{LASER}/tools-external/fastBPE/fast'
MOSES_BDIR = f'{LASER}/tools-external/moses-tokenizer/tokenizer/'
MOSES_TOKENIZER = f'{MOSES_BDIR}tokenizer.perl -q -no-escape -threads 20 -l '
MOSES_LC = f'{MOSES_BDIR}lowercase.perl'
NORM_PUNC = f'{MOSES_BDIR}normalize-punctuation.perl -l '
DESCAPE = f'{MOSES_BDIR}deescape-special-chars.perl'
REM_NON_PRINT_CHAR = f'{MOSES_BDIR}remove-non-printing-char.perl'

# Romanization (Greek only)
ROMAN_LC = f'python3 {LASER}/source/lib/romanize_lc.py -l '

# Mecab tokenizer for Japanese
MECAB = f'{LASER}/tools-external/mecab'


###############################################################################
#
# Tokenize a line of text
#
###############################################################################

def TokenLine(line, lang='en', lower_case=True, romanize=False):
    assert lower_case, 'lower case is needed by all the models'
    roman = lang if romanize else 'none'
    tok = check_output(
        (
            (
                (
                    (
                        REM_NON_PRINT_CHAR
                        + '|'
                        + NORM_PUNC
                        + lang
                        + '|'
                        + DESCAPE
                        + '|'
                        + MOSES_TOKENIZER
                        + lang
                        + ('| python3 -m jieba -d ' if lang == 'zh' else '')
                        + (
                            f'|{MECAB}/bin/mecab -O wakati -b 50000 '
                            if lang == 'ja'
                            else ''
                        )
                    )
                    + '|'
                )
                + ROMAN_LC
            )
            + roman
        ),
        input=line,
        encoding='UTF-8',
        shell=True,
    )
    return tok.strip()


###############################################################################
#
# Tokenize a file
#
###############################################################################

def Token(inp_fname, out_fname, lang='en',
          lower_case=True, romanize=False, descape=False,
          verbose=False, over_write=False, gzip=False):
    assert lower_case, 'lower case is needed by all the models'
    assert not over_write, 'over-write is not yet implemented'
    if not os.path.isfile(out_fname):
        cat = 'zcat ' if gzip else 'cat '
        roman = lang if romanize else 'none'
        # handle some iso3 langauge codes
        if lang in ('cmn', 'wuu', 'yue'):
            lang = 'zh'
        if lang in ('jpn'):
            lang = 'ja'
        if verbose:
            print(
                f" - Tokenizer: {os.path.basename(inp_fname)} in language {lang} {'(gzip)' if gzip else ''} {'(de-escaped)' if descape else ''}"
            )
        run(
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                cat
                                                + inp_fname
                                                + '|'
                                                + REM_NON_PRINT_CHAR
                                                + '|'
                                                + NORM_PUNC
                                                + lang
                                                + (
                                                    f'|{DESCAPE}'
                                                    if descape
                                                    else ''
                                                )
                                            )
                                            + '|'
                                        )
                                        + MOSES_TOKENIZER
                                    )
                                    + lang
                                )
                                + (
                                    '| python3 -m jieba -d '
                                    if lang == 'zh'
                                    else ''
                                )
                                + (
                                    f'|{MECAB}/bin/mecab -O wakati -b 50000 '
                                    if lang == 'ja'
                                    else ''
                                )
                                + '|'
                            )
                            + ROMAN_LC
                        )
                        + roman
                    )
                    + '>'
                )
                + out_fname
            ),
            env=dict(os.environ, LD_LIBRARY_PATH=f'{MECAB}/lib'),
            shell=True,
        )
    elif not over_write and verbose:
        print(f' - Tokenizer: {os.path.basename(out_fname)} exists already')


###############################################################################
#
# Apply FastBPE on one line of text
#
###############################################################################

def BPEfastLoad(line, bpe_codes):
    bpe_vocab = bpe_codes.replace('fcodes', 'fvocab')
    return fastBPE.fastBPE(bpe_codes, bpe_vocab)

def BPEfastApplyLine(line, bpe):
    return bpe.apply([line])[0]


###############################################################################
#
# Apply FastBPE on a whole file
#
###############################################################################

def BPEfastApply(inp_fname, out_fname, bpe_codes,
                 verbose=False, over_write=False):
    if not os.path.isfile(out_fname):
        if verbose:
            print(f' - fast BPE: processing {os.path.basename(inp_fname)}')
        bpe_vocab = bpe_codes.replace('fcodes', 'fvocab')
        if not os.path.isfile(bpe_vocab):
            print(f' - fast BPE: focab file not found {bpe_vocab}')
            bpe_vocab = ''
        run(
            f'{FASTBPE} applybpe {out_fname} {inp_fname} {bpe_codes} {bpe_vocab}',
            shell=True,
            stderr=DEVNULL,
        )
    elif not over_write and verbose:
        print(f' - fast BPE: {os.path.basename(out_fname)} exists already')


###############################################################################
#
# Split long lines into multiple sentences at "."
#
###############################################################################

def SplitLines(ifname, of_txt, of_sid):
    if os.path.isfile(of_txt):
        print(f' - SplitLines: {of_txt} already exists')
        return
    nl = 0
    nl_sp = 0
    maxw = 0
    maxw_sp = 0
    with open(of_sid, 'w') as fp_sid:
        fp_txt = open(of_txt, 'w')
        with open(ifname, 'r') as ifp:
            for line in ifp:
                print('{:d}'.format(nl), file=fp_sid)  # store current sentence ID
                nw = 0
                words = line.strip().split()
                maxw = max(maxw, len(words))
                for i, word in enumerate(words):
                    if word == '.' and i != len(words)-1:
                        if nw > 0:
                            print(f' {word}', file=fp_txt)
                        else:
                            print(f'{word}', file=fp_txt)
                        # store current sentence ID
                        print('{:d}'.format(nl), file=fp_sid)
                        nl_sp += 1
                        maxw_sp = max(maxw_sp, nw+1)
                        nw = 0
                    else:
                        if nw > 0:
                            print(f' {word}', end='', file=fp_txt)
                        else:
                            print(f'{word}', end='', file=fp_txt)
                        nw += 1
                if nw > 0:
                    # handle remainder of sentence
                    print('', file=fp_txt)
                    nl_sp += 1
                    maxw_sp = max(maxw_sp, nw+1)
                nl += 1
        print(f' - Split sentences: {ifname}')
        print(' -                  lines/max words: {:d}/{:d} -> {:d}/{:d}'
              .format(nl, maxw, nl_sp, maxw_sp))
    fp_txt.close()


###############################################################################
#
# Join embeddings of previously split lines (average)
#
###############################################################################

def JoinEmbed(if_embed, sid_fname, of_embed, dim=1024):
    if os.path.isfile(of_embed):
        print(f' - JoinEmbed: {of_embed} already exists')
        return
    # read the input embeddings
    em_in = np.fromfile(if_embed, dtype=np.float32, count=-1).reshape(-1, dim)
    ninp = em_in.shape[0]
    print(' - Combine embeddings:')
    print('                input: {:s} {:d} sentences'.format(if_embed, ninp))

    # get all sentence IDs
    sid = np.empty(ninp, dtype=np.int32)
    i = 0
    with open(sid_fname, 'r') as fp_sid:
        for line in fp_sid:
            sid[i] = int(line)
            i += 1
    nout = sid.max() + 1
    print('                IDs: {:s}, {:d} sentences'.format(sid_fname, nout))

    # combining
    em_out = np.zeros((nout, dim), dtype=np.float32)
    cnt = np.zeros(nout, dtype=np.int32)
    for i in range(ninp):
        idx = sid[i]
        em_out[idx] += em_in[i]  # cumulate sentence vectors
        cnt[idx] += 1

    if (cnt == 0).astype(int).sum() > 0:
        print('ERROR: missing lines')
        sys.exit(1)

    # normalize
    for i in range(nout):
        em_out[i] /= cnt[i]

    print('                output: {:s}'.format(of_embed))
    em_out.tofile(of_embed)
