# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import json
import rouge

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
# from sumy.summarizers.kl import KLSummarizer
from sumy.utils import get_stop_words

LANGUAGE = 'vietnamese'


def read_data_from_json(path_file):
    f = open(path_file, 'r', encoding='utf-8')
    data = json.load(f)
    f.close()
    return data


def prepare_results(metric, p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1',
                                                                 100.0 * f)


# all_hypothesis = [hypothesis_1, hypothesis_2]
# all_references = [references_1, references_2]
def get_rouge_score(all_hypothesis, all_references):
    for aggregator in ['Avg', 'Best', 'Individual']:
        print('Evaluation with {}'.format(aggregator))
        apply_avg = aggregator == 'Avg'
        apply_best = aggregator == 'Best'

        evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                                max_n=4,
                                limit_length=True,
                                length_limit=100,
                                length_limit_type='words',
                                apply_avg=apply_avg,
                                apply_best=apply_best,
                                alpha=0.5,  # Default F1_score
                                weight_factor=1.2,
                                stemming=True)

        scores = evaluator.get_scores(all_hypothesis, all_references)

        for metric, results in sorted(scores.items(), key=lambda x: x[0]):
            if not apply_avg and not apply_best:  # value is a type of list as we evaluate each summary vs each reference
                for hypothesis_id, results_per_ref in enumerate(results):
                    nb_references = len(results_per_ref['p'])
                    for reference_id in range(nb_references):
                        print('\tHypothesis #{} & Reference #{}: '.format(hypothesis_id, reference_id))
                        print('\t' + prepare_results(metric, results_per_ref['p'][reference_id],
                                                     results_per_ref['r'][reference_id],
                                                     results_per_ref['f'][reference_id]))
                print()
            else:
                print(prepare_results(metric, results['p'], results['r'], results['f']))
        print()


def lexrank_summary(text, sentences_count):
    parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))

    summarizer = LexRankSummarizer()
    summarizer.stop_words = get_stop_words(LANGUAGE)

    result = ' '
    for sentence in summarizer(parser.document, sentences_count):
        result += sentence._text
    return result.strip()


def get_documents(json_original):
    all_originnal = []
    for exam in json_original:
        context = exam['context']
        text = ''
        for doc in context:
            text += doc
        all_originnal.append(text.strip())
    return all_originnal


def get_summarys(json_summary):
    all_summary = []
    for exam in json_summary:
        summary_0 = exam['0_tokened.gold.txt']
        summary_1 = exam['1_tokened.gold.txt']
        length = len(summary_0) if (len(summary_0) > len(summary_1)) else len(summary_1)
        summary_0 = ' '.join(summary_0)
        summary_1 = ' '.join(summary_1)
        all_summary.append(([summary_0, summary_1], length))
    return all_summary


if __name__ == '__main__':
    json_original = read_data_from_json('./Data/original.json')
    json_summary = read_data_from_json('./Data/summary_tokend.json')

    all_originnal = get_documents(json_original)
    all_summary = get_summarys(json_summary)
    all_machine_summary = []
    all_gold_summary = []

    i = 0
    while i < len(all_originnal):
        try:
            summary = lexrank_summary(all_originnal[i].replace('\n', ' '), all_summary[i][1])
            all_machine_summary.append(summary)
            all_gold_summary = all_summary[i][0]
            print(i)
            i += 1
        except:
            print(i, 'Lỗi')
            continue

    get_rouge_score(all_machine_summary, all_gold_summary)
