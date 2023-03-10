""" Translation main class """
from __future__ import unicode_literals, print_function

import torch
from onmt.inputters.text_dataset import TextMultiField


class TranslationBuilder(object):
    """
    Build a word-based translation from the batch output
    of translator and the underlying dictionaries.

    Replacement based on "Addressing the Rare Word
    Problem in Neural Machine Translation" :cite:`Luong2015b`

    Args:
       data (onmt.inputters.Dataset): Data.
       fields (List[Tuple[str, torchtext.data.Field]]): data fields
       n_best (int): number of translations produced
       replace_unk (bool): replace unknown words using attention
       has_tgt (bool): will the batch have gold targets
    """

    def __init__(self, data, fields, n_best=1, replace_unk=False,
                 has_tgt=False, phrase_table=""):
        self.data = data
        self.fields = fields
        self._has_text_src = isinstance(
            dict(self.fields)["src"], TextMultiField)
        self.n_best = n_best
        self.replace_unk = replace_unk
        self.phrase_table = phrase_table
        self.has_tgt = has_tgt

    def _build_target_tokens(self, src, src_vocab, src_raw, pred, attn):
        tgt_field = dict(self.fields)["tgt"].base_field
        src_field = dict(self.fields)["src"].base_field


        vocab = tgt_field.vocab
        tokens = []

        # print(src)
        # print(src_raw)
        # print('***************')
        # print(pred)

        for tok in pred:
            if tok < len(vocab):
                tokens.append(vocab.itos[tok])
            else:
                tokens.append(src_vocab.itos[tok - len(vocab)])
            if tokens[-1] == tgt_field.eos_token:
                tokens = tokens[:-1]
                break
        if self.replace_unk and attn is not None and src is not None:
            src_vocab = src_field.vocab

            src_unks = [
                (i, src_token)
                for i, (src_token, src_id) in enumerate(zip(src_raw, src))
                if src_vocab.itos[src_id] == src_field.unk_token
            ]
            seen_tgt_unks = 0
            total_src_unks = len(src_unks)

            # print('---------Start----------')
            # print(src_raw)
            # print(src_unks)
            # print(total_src_unks)

            pred_mask = [True for _ in range(len(tokens))]

            for i in range(len(tokens)):
                if tokens[i] == tgt_field.unk_token:
                    if seen_tgt_unks < total_src_unks:
                        tokens[i] = src_unks[seen_tgt_unks][1]
                    else:
                        pred_mask[i] = False
                    seen_tgt_unks += 1

                    # if seen_tgt_unks < total_src_unks:
                    #     _, max_index = attn[i].index_select(0, torch.LongTensor([unk[0] for unk in src_unks]).cuda()).max(0)
                    #     tokens[i] = src_raw[src_unks[max_index.item()][0]]
                    #     del src_unks[max_index.item()]
                    # else:
                    #     pred_mask[i] = False
                    # seen_tgt_unks += 1

                    # _, max_index = attn[i][:len(src_raw)].max(0)
                    # tokens[i] = src_raw[max_index.item()]
                    # if self.phrase_table != "":
                    #     with open(self.phrase_table, "r") as f:
                    #         for line in f:
                    #             if line.startswith(src_raw[max_index.item()]):
                    #                 tokens[i] = line.split('|||')[1].strip()

                # print(tokens)
                # print(seen_tgt_unks)
                # print('--------End----------')

        return [token for token, mask in zip(tokens, pred_mask) if mask]

    def from_batch(self, translation_batch):
        batch = translation_batch["batch"]
        assert(len(translation_batch["gold_score"]) ==
               len(translation_batch["predictions"]))
        batch_size = batch.batch_size

        preds, pred_score, attn, gold_score, indices = list(zip(
            *sorted(zip(translation_batch["predictions"],
                        translation_batch["scores"],
                        translation_batch["attention"],
                        translation_batch["gold_score"],
                        batch.indices.data),
                    key=lambda x: x[-1])))

        # Sorting
        inds, perm = torch.sort(batch.indices)
        if self._has_text_src:
            src = batch.src[0][:, :, 0].index_select(1, perm)
        else:
            src = None
        tgt = batch.tgt[:, :, 0].index_select(1, perm) \
            if self.has_tgt else None

        translations = []
        for b in range(batch_size):
            if self._has_text_src:
                src_vocab = self.data.src_vocabs[inds[b]] \
                    if self.data.src_vocabs else None
                src_raw = self.data.examples[inds[b]].src[0]
            else:
                src_vocab = None
                src_raw = None
            pred_sents = [self._build_target_tokens(
                src[:, b] if src is not None else None,
                src_vocab, src_raw,
                preds[b][n], attn[b][n])
                for n in range(self.n_best)]
            gold_sent = None
            if tgt is not None:
                gold_sent = self._build_target_tokens(
                    src[:, b] if src is not None else None,
                    src_vocab, src_raw,
                    tgt[1:, b] if tgt is not None else None, None)

            translation = Translation(
                src[:, b] if src is not None else None,
                src_raw, pred_sents, attn[b], pred_score[b],
                gold_sent, gold_score[b]
            )
            translations.append(translation)

        return translations


class Translation(object):
    """Container for a translated sentence.

    Attributes:
        src (LongTensor): Source word IDs.
        src_raw (List[str]): Raw source words.
        pred_sents (List[List[str]]): Words from the n-best translations.
        pred_scores (List[List[float]]): Log-probs of n-best translations.
        attns (List[FloatTensor]) : Attention distribution for each
            translation.
        gold_sent (List[str]): Words from gold translation.
        gold_score (List[float]): Log-prob of gold translation.
    """

    __slots__ = ["src", "src_raw", "pred_sents", "attns", "pred_scores",
                 "gold_sent", "gold_score"]

    def __init__(self, src, src_raw, pred_sents,
                 attn, pred_scores, tgt_sent, gold_score):
        self.src = src
        self.src_raw = src_raw
        self.pred_sents = pred_sents
        self.attns = attn
        self.pred_scores = pred_scores
        self.gold_sent = tgt_sent
        self.gold_score = gold_score

    def log(self, sent_number):
        """
        Log translation.
        """

        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        pred_sent = ' '.join(best_pred)
        msg = [
            f'\nSENT {sent_number}: {self.src_raw}\n',
            *(
                f'PRED {sent_number}: {pred_sent}\n',
                "PRED SCORE: {:.4f}\n".format(best_score),
            ),
        ]
        if self.gold_sent is not None:
            tgt_sent = ' '.join(self.gold_sent)
            msg.extend(
                (
                    f'GOLD {sent_number}: {tgt_sent}\n',
                    "GOLD SCORE: {:.4f}\n".format(self.gold_score),
                )
            )
        if len(self.pred_sents) > 1:
            msg.append('\nBEST HYP:\n')
            msg.extend(
                "[{:.4f}] {}\n".format(score, sent)
                for score, sent in zip(self.pred_scores, self.pred_sents)
            )
        return "".join(msg)
