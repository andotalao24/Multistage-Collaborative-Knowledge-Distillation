import argparse
import collections
import copy
import json
import os
import random

import nltk
import numpy as np
from tqdm import tqdm
from utils import read_row,store_row

word_tags = set(['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN',
                    'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS',
                    'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
                    'WDT', 'WP', 'WP$', 'WRB'])


def is_word(tag):
    return tag in word_tags


def preprocess_parsed(tr):
    assert isinstance(tr, nltk.Tree), (type(tr), tr)
    def func(tr):
        if len(tr) == 1 and isinstance(tr[0], str):
            return [f'( {tr[0]} )']
        nodes = []
        for x in tr:
            this_nodes = func(x)
            nodes.extend(this_nodes)
        if len(nodes) > 1:
            nodes = ['('] + nodes + [')']
        return nodes
    nodes = func(tr)
    return ' '.join(nodes)


def remove_words_by_mask_keep_labels(tr, mask):
    kept, removed = [], []
    def func(tr, pos=0):
        if not isinstance(tr, (list, tuple)):
            if mask[pos] == False:
                removed.append(tr)
                return None, 1
            kept.append(tr)
            return tr, 1

        size = 0
        node = []

        for subtree in tr:
            x, xsize = func(subtree, pos=pos + size)
            if x is not None:
                node.append(x)
            size += xsize

        for x in node:
            if isinstance(x, (list, tuple)):
                assert len(x) > 1

        if len(node) == 1:
            node = node[0]
        elif len(node) == 0:
            return None, size
        if isinstance(node, (list, tuple)):
            node = f'({tr.label().strip()} {" ".join(node)})'
        else:
            node = f'({tr.label().strip()} {node})'
        return node, size
    new_tree, _ = func(tr)
    return new_tree, kept, removed


def remove_words_by_mask(tr, mask):
    kept, removed = [], []
    def func(tr, pos=0):
        if not isinstance(tr, (list, tuple)):
            if mask[pos] == False:
                removed.append(tr)
                return None, 1
            kept.append(tr)
            return tr, 1

        size = 0
        node = []

        for subtree in tr:
            x, xsize = func(subtree, pos=pos + size)
            if x is not None:
                node.append(x)
            size += xsize

        for x in node:
            if isinstance(x, (list, tuple)):
                assert len(x) > 1

        if len(node) == 1:
            node = node[0]
        elif len(node) == 0:
            return None, size
        if isinstance(node, list):
            node = tuple(node)
        return node, size
    new_tree, _ = func(tr)
    return new_tree, kept, removed


def flatten_tree(tr):
    def func(tr):
        if not isinstance(tr, (list, tuple)):
            return [tr]
        result = []
        for x in tr:
            result += func(x)
        return result
    return func(tr)


def parsing_f1(gold_spans_set, prediction_spans_set):
    correct = len(gold_spans_set.intersection(prediction_spans_set))
    if correct == 0:
        return 0., 0., 0.
    gold_total = len(gold_spans_set)
    prediction_total = len(prediction_spans_set)
    if gold_total == 0 or prediction_total == 0:
        return 0., 0., 0.
    prec = float(correct) / prediction_total
    recall = float(correct) / gold_total
    f1 = 2 * (prec * recall) / (prec + recall)
    return f1, prec, recall


def get_spans_w_labels(tree):
    assert isinstance(tree, nltk.Tree)
    spans = []

    def helper(tr, pos=0):
        if isinstance(tr, (str, int)):
            return 1
        child_labels = []
        size = 0
        for x in tr:
            xsize = helper(x, pos + size)
            size += xsize
            if xsize > 1:
                child_labels.append(x.label())
            else:
                child_labels.append('-')
        if size > 1:
            spans.append((tr.label(), tuple(child_labels), pos, size))
        return size

    length = helper(tree)

    return spans, length


def get_spans(tree):
    #assert not isinstance(tree, nltk.Tree)
    #assert isinstance(tree, (list, tuple)), tree
    spans = []

    def helper(tr, pos=0):
        if isinstance(tr, (str, int)):
            return 1
        size = 0
        for x in tr:
            xsize = helper(x, pos + size)
            size += xsize
        if size > 1:
            spans.append((pos, size))
        return size

    length = helper(tree)

    return spans, length


def get_spans_set(tree):
    spans, length = get_spans(tree)
    spans = set(spans)
    return spans


def preprocess_punctuation(tokens, ref_tree_string, tree_string, has_tags=True, has_labels=False):
    """ Remove punctuation from tree and sentence. """
    if not has_tags:
        tree_string = tree_string.replace('(', '(X ')
    ref_nltk_tree = nltk.Tree.fromstring(ref_tree_string)
    nltk_tree = nltk.Tree.fromstring(tree_string)
    tags = [x[1] for x in ref_nltk_tree.pos()]
    #assert len(tokens) == len(tags), (tokens, tags, ref_tree_string, tree_string)

    # Update sentence and tree after removing punctuation.
    is_word_mask = [is_word(tag) for tag in tags]
    new_tree, kept_nodes, removed_nodes = remove_words_by_mask(nltk_tree, is_word_mask)
    new_tree_w_labels = None
    if has_labels:
        new_tree_w_labels, _, _ = remove_words_by_mask_keep_labels(nltk_tree, is_word_mask)
    new_tokens = [w for w, t in zip(tokens, is_word_mask) if t]

    # Sanity check.
    check_new_tokens = flatten_tree(new_tree)
    if len(new_tokens) != len(check_new_tokens):
        missing = [x for x in new_tokens if x not in check_new_tokens]
        wrong = [x for x in check_new_tokens if x not in new_tokens]
        #print(f'MISMATCH {new_tokens} != {check_new_tokens}, missing = {missing}, wrong = {wrong}')
        raise ValueError

    return new_tokens, new_tree, new_tree_w_labels


def fixup_prediction(unparsed, tree_string):
    # Add missing brackets.
    while tree_string.count('(') > tree_string.count(')'):
        tree_string = tree_string + ' )'

    # Remove extra brackets.
    while tree_string.count(')') > tree_string.count('(') and tree_string[-2:] == ' )':
        tree_string = tree_string[:-2]

    # Separate merged tokens (by /).
    tokens = []
    for tok in tree_string.split():
        if len(tok) > 1 and '/' in tok and ' / '.join(tok.split('/')) in unparsed:
            for i, subtok in enumerate(tok.split('/')):
                if i > 0:
                    tokens.append(f'( / )')
                tokens.append(f'( {subtok} )')
        else:
            tokens.append(tok)
    tree_string = ' '.join(tokens)

    # Separate merged tokens (by -).
    tokens = []
    for tok in tree_string.split():
        if len(tok) > 1 and '-' in tok and ' - '.join(tok.split('-')) in unparsed:
            for i, subtok in enumerate(tok.split('-')):
                if i > 0:
                    tokens.append(f'( - )')
                tokens.append(f'( {subtok} )')
        else:
            tokens.append(tok)
    tree_string = ' '.join(tokens)

    # Separate other merged tokens.
    tokens = []
    for tok in tree_string.split():
        if tok == "won't" and "wo n't" in unparsed:
            tokens.append("( wo n't )")
        elif tok == "don't" and "do n't" in unparsed:
            tokens.append("( do n't )")
        elif tok == "can't" and "ca n't" in unparsed:
            tokens.append("( ca n't )")
        elif tok.endswith("'s") and tok not in unparsed:
            subtok = tok[:-len("'s")]
            tokens.append(f"( {subtok} 's )")
        else:
            tokens.append(tok)
    tree_string = ' '.join(tokens)

    # Fix spacing. For example in "('s"
    tree_string = ' '.join(tree_string.replace('(', '( ').split())

    return tree_string


def run_eval(unparsed, parsed, raw_prediction, with_labels=False):
    obj = {}
    # Process.
    tokens = unparsed.strip().split()
    tree_string = parsed.strip()
    new_tokens, new_tree, new_tree_w_labels = preprocess_punctuation(tokens, tree_string, tree_string, has_labels=with_labels)
    obj['new_tokens'] = new_tokens
    obj['new_tree'] = new_tree
    obj['new_tree_w_labels'] = new_tree_w_labels

    fixed_prediction = fixup_prediction(unparsed, raw_prediction)
    try:
        _, new_tree_prediction, _ = preprocess_punctuation(tokens, tree_string, fixed_prediction, has_tags=False)
        obj['new_tree_prediction'] = new_tree_prediction

        flat_tree = flatten_tree(new_tree)
        flat_tree_prediction = flatten_tree(new_tree_prediction)

        if len(flat_tree) != len(flat_tree_prediction):
            obj['success'] = False
            error_type = 'token count after remove punct'
            obj['error_type'] = error_type
        else:
            obj['success'] = True
    except ValueError:
        error_type = 'fail to parse'
        #print(f'ERROR: {error_type}.\nGOLD = {new_tree}\nPRED = {fixed_prediction}\nPRAW = {raw_prediction}')
        obj['success'] = False
        obj['error_type'] = error_type
    except IndexError:
        # TODO: Not sure why this happens...
        error_type = 'fail to parse [IndexError]'
        #print(f'ERROR: {error_type}.\nGOLD = {new_tree}\nPRED = {fixed_prediction}')
        obj['success'] = False
        obj['error_type'] = error_type

    # Evaluate.
    f1, precision, recall = 0., 0., 0.
    if obj['success']:
        gold_spans = get_spans_set(obj['new_tree'])
        prediction_spans = get_spans_set(obj['new_tree_prediction'])
        f1, precision, recall = parsing_f1(gold_spans, prediction_spans)
    obj['f1'], obj['precision'], obj['recall'] = f1, precision, recall

    # Evaluate w/ labels.
    def safe_label(label):
        if label == '-':
            return label
        return label.split('-')[0].split('=')[0]
    if with_labels:
        found_labeled_span = []
        missed_labeled_span = []
        found_labeled_subtr = []
        missed_labeled_subtr = []
        if obj['success']:
            gold_spans, _ = get_spans_w_labels(nltk.Tree.fromstring(obj['new_tree_w_labels']))
            prediction_spans = get_spans_set(obj['new_tree_prediction'])
            for label, child_labels, start, end in gold_spans:
                child_labels = ' '.join([safe_label(y) for y in child_labels])
                label = safe_label(label)
                subtr = f'{label} -> {child_labels}'
                if (start, end) in prediction_spans:
                    found_labeled_span.append((label, start, end))
                else:
                    missed_labeled_span.append((label, start, end))
                if (start, end) in prediction_spans:
                    found_labeled_subtr.append((subtr, start, end))
                else:
                    missed_labeled_subtr.append((subtr, start, end))
        else:
            gold_spans, _ = get_spans_w_labels(nltk.Tree.fromstring(obj['new_tree_w_labels']))
            for label, child_labels, start, end in gold_spans:
                child_labels = ' '.join([safe_label(y) for y in child_labels])
                label = safe_label(label)
                subtr = f'{label} -> {child_labels}'
                missed_labeled_span.append((label, start, end))
                missed_labeled_subtr.append((subtr, start, end))

        obj['found_labeled_span'] = found_labeled_span
        obj['missed_labeled_span'] = missed_labeled_span
        obj['found_labeled_subtr'] = found_labeled_subtr
        obj['missed_labeled_subtr'] = missed_labeled_subtr

    return obj

def agreeTest(unparsed, parsed, teacher_pred, student_pred):
    assert len(teacher_pred) == len(student_pred)
    l= len(teacher_pred)
    # Teacher
    teacher_eval_objs = []
    student_eval_objs = []
    for i in range(l):

        sent = unparsed[i]
        gold_w_labels = parsed[i]
        gold = preprocess_parsed(nltk.Tree.fromstring(gold_w_labels))
        pred = teacher_pred[i]
        eval_obj = run_eval(sent, gold_w_labels, pred, with_labels=True)
        teacher_eval_objs.append(eval_obj)

        pred = student_pred[i]
        eval_obj = run_eval(sent, gold_w_labels, pred, with_labels=True)
        student_eval_objs.append(eval_obj)

    # Agreement
    agreement_eval_objs = []
    f1s=[]
    for t, s in zip(teacher_eval_objs, student_eval_objs):
        obj = {}
        f1, precision, recall = 0., 0., 0.
        obj['success'] = t['success'] and s['success']
        if obj['success']:
            t_spans = get_spans_set(t['new_tree_prediction'])
            s_spans = get_spans_set(s['new_tree_prediction'])
            f1, precision, recall = parsing_f1(t_spans, s_spans)
        obj['f1'], obj['precision'], obj['recall'] = f1, precision, recall
        f1s.append(f1)
    return f1s


def getF1(preds,golds,unparse,has_tag=False):
    objs=[]
    i=0
    f1s=[]
    for pred,gold in zip(preds,golds):
        unparsed=unparse[i]
        i+=1
        #if len(unparsed.split())<=2:
            #continue
        try:
            obj = run_eval(unparsed,gold,pred,has_tag)
            f1=obj['f1']
            f1s.append(f1)
        except Exception as e:
            print(e)
            f1=-1
            f1s.append(f1)
            continue
        objs.append(obj)
    #f1s = [obj['f1'] for obj in objs]
    f1s_success = [obj['f1'] for obj in objs if obj['success']]
    return f1s_success,f1s


def main():
    corpus = []
    with open(args.ptb_file) as f:
        for line in f:
            corpus.append(json.loads(line))

    pred_corpus = []
    with open(args.pred_file) as f:
        for line in f:
            pred_corpus.append(json.loads(line))

    if args.mode in ('teacher', 'student'):
        if args.mode == 'student':
            assert len(corpus) == len(pred_corpus)
            for ex, pred in zip(corpus, pred_corpus):
                ex['translation']['unlabeled'] = pred

        eval_objs = []
        for ex in tqdm(corpus):
            ex = ex['translation']
            sent = ex['unparsed']
            gold_w_labels = ex['parsed']
            gold = preprocess_parsed(nltk.Tree.fromstring(gold_w_labels))
            pred = ex['unlabeled']
            eval_obj = run_eval(sent, gold_w_labels, pred, with_labels=True)
            eval_objs.append(eval_obj)

        print(f'{len(eval_objs)} {np.mean([x["f1"] for x in eval_objs]):.3f}')
        print(f'success {len([x for x in eval_objs if x["success"]])}')
        print('')

        print('LABEL RECALL')
        correct = collections.Counter()
        total = collections.Counter()
        for x in eval_objs:
            for label, start, end in x['found_labeled_span']:
                correct[label] += 1
                total[label] += 1
            for label, start, end in x['missed_labeled_span']:
                total[label] += 1
        for k in sorted(total.keys()):
            num = total[k]
            num_correct = correct[k]
            print(f'{k} {num_correct} / {num} ({num_correct/num:.3f})')
        print('')

        if args.subtr:
            print('SUBTR RECALL')
            correct = collections.Counter()
            total = collections.Counter()
            for x in eval_objs:
                for label, start, end in x['found_labeled_subtr']:
                    correct[label] += 1
                    total[label] += 1
                for label, start, end in x['missed_labeled_subtr']:
                    total[label] += 1
            for k in sorted(total.keys()):
                num = total[k]
                num_correct = correct[k]
                print(f'{k} {num_correct} / {num} ({num_correct/num:.3f})')

    elif args.mode == 'agreement':
        # Teacher
        teacher_eval_objs = []
        for ex in tqdm(corpus):
            ex = ex['translation']
            sent = ex['unparsed']
            gold_w_labels = ex['parsed']
            gold = preprocess_parsed(nltk.Tree.fromstring(gold_w_labels))
            pred = ex['unlabeled']
            eval_obj = run_eval(sent, gold_w_labels, pred, with_labels=True)
            teacher_eval_objs.append(eval_obj)

        # Student
        assert len(corpus) == len(pred_corpus)
        for ex, pred in zip(corpus, pred_corpus):
            ex['translation']['unlabeled'] = pred['translation']['unlabeled']

        student_eval_objs = []
        for ex in tqdm(corpus):
            ex = ex['translation']
            sent = ex['unparsed']
            gold_w_labels = ex['parsed']
            gold = preprocess_parsed(nltk.Tree.fromstring(gold_w_labels))
            pred = ex['unlabeled']
            eval_obj = run_eval(sent, gold_w_labels, pred, with_labels=True)
            student_eval_objs.append(eval_obj)

        # Agreement
        agreement_eval_objs = []
        for t, s in zip(teacher_eval_objs, student_eval_objs):
            obj = {}
            f1, precision, recall = 0., 0., 0.
            obj['success'] = t['success'] and s['success']
            if obj['success']:
                t_spans = get_spans_set(t['new_tree_prediction'])
                s_spans = get_spans_set(s['new_tree_prediction'])
                f1, precision, recall = parsing_f1(t_spans, s_spans)
            obj['f1'], obj['precision'], obj['recall'] = f1, precision, recall
            agreement_eval_objs.append(obj)

        eval_objs = agreement_eval_objs
        s_f1 = np.mean([x['f1'] for x in student_eval_objs])
        t_f1 = np.mean([x['f1'] for x in teacher_eval_objs])
        print('Agreement')
        print(f'{len(eval_objs)} {np.mean([x["f1"] for x in eval_objs]):.3f}')
        print(f'success {len([x for x in eval_objs if x["success"]])}')
        print(f'T={t_f1:.3f} S={s_f1:.3f}')
        print('')

        ## Analysis
        f1s = [x['f1'] for x in agreement_eval_objs]
        index = np.argsort(f1s)
        for i, subindex in enumerate(np.array_split(index, 10)):
            t_f1s = [teacher_eval_objs[ix]['f1'] for ix in subindex]
            s_f1s = [student_eval_objs[ix]['f1'] for ix in subindex]
            a_f1s = [agreement_eval_objs[ix]['f1'] for ix in subindex]

            k = 'precision'
            t_prec = np.mean([teacher_eval_objs[ix][k] for ix in subindex])
            s_prec = np.mean([student_eval_objs[ix][k] for ix in subindex])

            k = 'recall'
            t_rec = np.mean([teacher_eval_objs[ix][k] for ix in subindex])
            s_rec = np.mean([student_eval_objs[ix][k] for ix in subindex])

            s_better = sum([1 if s_f1 > t_f1 else 0 for t_f1, s_f1 in zip(t_f1s, s_f1s)])
            t_better = sum([1 if s_f1 < t_f1 else 0 for t_f1, s_f1 in zip(t_f1s, s_f1s)])
            same = len(subindex) - s_better - t_better

            print(f'{i} A={np.mean(a_f1s):.3f} T={np.mean(t_f1s):.3f} ({t_prec:.3f},{t_rec:.3f}) S={np.mean(s_f1s):.3f} ({s_prec:.3f},{s_rec:.3f}) S+,T+,X={s_better},{t_better},{same}')

        agreement_file = f'{args.pred_file}.agreement'
        print(f'writing {agreement_file}')
        with open(agreement_file, 'w') as f:
            for x in agreement_eval_objs:
                f.write(f'{x["f1"]}\n')

        if False:
            for a, t, s in zip(agreement_eval_objs, teacher_eval_objs, student_eval_objs):
                print(f"{a['f1']},{t['f1']},{s['f1']}")

        ## More Analysis

        if False:
            for ex, t, s in zip(corpus, teacher_eval_objs, student_eval_objs):
                ex = ex['translation']
                sent = ex['unparsed']
                gold_w_labels = ex['parsed']
                gold = nltk.Tree.fromstring(gold_w_labels)

                t_f1, s_f1 = t['f1'], s['f1']
                if t_f1 < s_f1:
                    gold.pretty_print()
                    print(f'T={t["new_tree_prediction"]}')
                    print(f'S={s["new_tree_prediction"]}')
                    print(f'T={t_f1:.3f}, S={s_f1:.3f}, DIFF={s_f1-t_f1:.3f}')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--craftfile', default='data_new/craft/silver-craft-chat-50-all.json', type=str)
    parser.add_argument('--ptbfile',default='data_new/chatgpt-silver-train-ptb-m50.json',type=str)
    parser.add_argument('--pred-file', default='transformers_output/chatgpt-m250.first_student.t5_small.id001-safe_valid/generated_predictions.txt', type=str)
    parser.add_argument('--subtr', action='store_true')
    parser.add_argument('--mode', choices=('teacher', 'student', 'agreement'), default='teacher')
    parser.add_argument('--preset', default=None, type=str)
    args = parser.parse_args()


    pseudo=read_row('data_new/silver-ptb-chat-50-all-5k-2.json')
    parsed_valid=[d['translation']['parsed'] for d in pseudo]
    unparsed_valid=[d['translation']['unparsed'] for d in pseudo]
    preds=[d['translation']['unlabeled'] for d in pseudo]
    f1s_success, f1s = getF1_new(preds, parsed_valid, unparsed_valid, True)
    avg = np.mean(np.array(f1s))
    f1Arr = f1s
    for i in range(len(pseudo)):
        pseudo[i]['translation']['f1'] = f1Arr[i]
    store_row('data_new/silver-ptb-chat-50-all-5k-2.json',pseudo)
    print('######evaluation: avg f1 is {}'.format(avg))

