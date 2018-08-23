#!/usr/bin/python

import os
import glob
import copy
import re
import logging
import click
from typing import Dict, Tuple, List


def read_text_file(filename: str) -> (str, int):
    output = ""
    with open(filename) as f:
        for line in f:
            output += line
    return output, len(output)


def read_a1_file(
    filename: str
) -> Tuple[Dict[str, Tuple[str, int, int]], Dict[int, str]]:
    events_in_text = {}
    span_annotations = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            cur_id, exp, _ = line.split("\t")

            if re.search(r"^T", cur_id):
                cur_type, begin, end = exp.split(" ")
                begin = int(begin)
                end = int(end)
                for i in range(begin, end):
                    events_in_text[i] = "E"
                span_annotations[cur_id] = [cur_type, begin, end]
            else:
                logging.warning("invalid annotation in a1 file: " + line)
    return span_annotations, events_in_text


def read_a2_file(
    filename: str,
    events_in_text_input: Dict[int, str],
    equiv_input: Dict[str, str],
    mode: str,
) -> Tuple[
    Dict[str, Tuple[str, int, int]],
    Dict[str, Tuple[str, str, List[str]]],
    Dict[str, int],
    Dict[str, int],
    Dict[int, str],
    Dict[str, str],
]:  # mode: either G for gold or A for predicted answer
    events_in_text = copy.deepcopy(events_in_text_input)
    equiv = copy.deepcopy(equiv_input)
    span_annotations = {}
    frame_annotations = {}
    span_list = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            line_parts = line.split("\t")
            cur_id = line_parts[0]
            exp = line_parts[1]

            if re.search(r"^T", cur_id):
                cur_type, begin, end = exp.split(" ")
                begin = int(begin)
                end = int(end)
                if mode == "G":
                    for i in range(begin, end):
                        events_in_text[i] = "E"
                span_annotations[cur_id] = [cur_type, begin, end]
                span_list.append([cur_id, cur_type, begin, end])
            elif re.search(r"^E", cur_id):
                exp_splitted = exp.split(" ")
                t_type, t_id = exp_splitted.pop(0).split(":")
                new_args = []
                for e_item in exp_splitted:
                    if e_item == "":
                        continue
                    a_type, a_id = e_item.split(":")
                    a_type = re.sub(r"^target[2-6]$", "target", a_type)
                    if a_id in equiv:
                        a_id = equiv[a_id]
                    new_args.append(a_type + ":" + a_id)
                frame_annotations[cur_id] = [t_type, t_id, new_args]
            elif re.search(r"^M", cur_id):
                cur_type, aid = exp.split(" ")
                frame_annotations[cur_id] = [cur_type, " ", ["target:" + aid]]
            elif re.search(r"^\*", cur_id):
                exp_splitted = exp.split(" ")
                rel = exp_splitted[0]
                pid = exp_splitted[1:]
                rep = pid[0]
                other = pid[1:]
                for o in other:
                    equiv[o] = rep
    e_list = []
    for key in frame_annotations:
        e_list.append(key)

    # detect and remove duplication by Equiv
    if mode == "A":
        # sort events
        new_e_list = []
        added = {}
        remain = []
        for e_item in e_list:
            remain.append(e_item)
        while len(remain) > 0:
            changep = 0
            for r in remain:
                e_arg = []
                for item in frame_annotations[r][2]:
                    if re.search(r"\:E[0-9-]+$", item):
                        e_arg.append(item)
                e_aid = [
                    parts[1] for e_arg_item in e_arg for parts in e_arg_item.split(":")
                ]
                danglingp = 0
                for e_a in e_aid:
                    if e_a not in added:
                        danglingp = 1
                        break
                if danglingp == 0:
                    new_e_list.append(r)
                    added[r] = 1
                    remain.remove(r)
                    changep = 1
            if changep == 0:
                logging.info(
                    "circular reference: [" + filename + "]" + ", ".join(remain)
                )
                new_e_list.extend(remain)
                remain = []

        e_list = new_e_list

        eventexp = {}  # for checking event duplication
        to_remove = []
        for e_id in e_list:
            # get event expression
            for r in frame_annotations[e_id][2]:
                if not re.search(r"\:", r):
                    continue
                a_type, a_id = r.split(":")
                if a_id in equiv:
                    a_id = equiv[a_id]
                r = a_type + ":" + a_id

            eventexp_key = frame_annotations[e_id][0] + "," + frame_annotations[e_id][1]
            eventexp_key += "," + ",".join(frame_annotations[e_id][2])

            # check duplication
            if eventexp_key in eventexp:
                d_id = eventexp[eventexp_key]
                del frame_annotations[e_id]
                to_remove.append(e_id)
                equiv[e_id] = d_id
                logging.info(
                    "["
                    + filename
                    + "]"
                    + e_id
                    + " is equivalent to "
                    + d_id
                    + " => removed."
                )
            else:
                eventexp[eventexp_key] = e_id

        for e_id in to_remove:
            e_list.remove(e_id)

    # get statistics
    num_event = {}
    for e_id in e_list:
        cur_type = frame_annotations[e_id][0]
        if cur_type not in num_event:
            num_event[cur_type] = 0
        num_event[cur_type] += 1

    num_span = {}
    for span in span_list:
        cur_type = span[1]
        if cur_type not in num_span:
            num_span[cur_type] = 0
        num_span[cur_type] += 1

    return (
        span_annotations,
        frame_annotations,
        num_event,
        num_span,
        events_in_text,
        equiv,
    )


def get_scores(gold: int, match_gold: int, answer: int, match_answer: int) -> float:
    precision = 0.0
    if answer > 0:
        precision = float(match_answer) / answer
    recall = 0.0
    if gold > 0:
        recall = float(match_gold) / gold
    f1 = 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return precision * 100, recall * 100, f1 * 100


def report_headline() -> None:
    print(
        "Class".ljust(20)
        + "\t"
        + "gold (match)".ljust(10)
        + "\t"
        + "answer (match)".ljust(10)
        + "\t"
        + "recall \t prec. \t fscore"
    )


def report(
    cl: str, gold: int, matched_gold: int, answer: int, matched_answer: int
) -> None:
    precision, recall, f1 = get_scores(gold, matched_gold, answer, matched_answer)
    gold_column = str(gold) + " (" + str(matched_gold) + ")"
    answer_column = str(answer) + " (" + str(matched_answer) + ")"
    print(
        cl.ljust(20)
        + "\t"
        + gold_column.ljust(10)
        + "\t"
        + answer_column.ljust(10)
        + "\t"
        + str(round(recall, 2))
        + "\t"
        + str(round(precision, 2))
        + "\t"
        + str(round(f1, 2))
    )


# METHODS FOR CHECKING EQUALITY OF SPANS AND EVENTS


def eq_event(
    aid: str,
    gid: str,
    a1_annotations: Dict[str, Tuple[str, int, int]],
    answers_span: Dict[str, Tuple[str, int, int]],
    answers_frame: Dict[str, Tuple[str, str, List[str]]],
    golds_span: Dict[str, Tuple[str, int, int]],
    golds_frame: Dict[str, Tuple[str, str, List[str]]],
    text: str,
    events_in_text: Dict[int, str],
    text_len: int,
    do_soft_class: bool,
    do_soft_args: bool,
    do_soft_span: bool,
    do_soft_overlap_span: bool,
) -> bool:
    if re.search(r"^E", aid):
        if (
            eq_class(
                aid,
                gid,
                a1_annotations,
                answers_span,
                answers_frame,
                golds_span,
                golds_frame,
                do_soft_class,
            )
            and eq_span(
                aid,
                gid,
                a1_annotations,
                answers_span,
                answers_frame,
                golds_span,
                golds_frame,
                text,
                events_in_text,
                text_len,
                do_soft_span,
                do_soft_overlap_span,
            )
            and eq_args(
                aid,
                gid,
                a1_annotations,
                answers_span,
                answers_frame,
                golds_span,
                golds_frame,
                text,
                events_in_text,
                text_len,
                do_soft_class,
                do_soft_args,
                do_soft_span,
                do_soft_overlap_span,
            )
        ):
            return True
    elif re.search(r"^M", aid):
        if eq_class(
            aid,
            gid,
            a1_annotations,
            answers_span,
            answers_frame,
            golds_span,
            golds_frame,
            do_soft_class,
        ) and eq_args(
            aid,
            gid,
            a1_annotations,
            answers_span,
            answers_frame,
            golds_span,
            golds_frame,
            text,
            events_in_text,
            text_len,
            do_soft_class,
            do_soft_args,
            do_soft_span,
            do_soft_overlap_span,
        ):
            return True
    else:
        return False


def eq_class(
    aid: str,
    gid: str,
    a1_annotations: Dict[str, Tuple[str, int, int]],
    answers_span: Dict[str, Tuple[str, int, int]],
    answers_frame: Dict[str, Tuple[str, str, List[str]]],
    golds_span: Dict[str, Tuple[str, int, int]],
    golds_frame: Dict[str, Tuple[str, str, List[str]]],
    do_soft_class: bool,
) -> bool:
    if aid in a1_annotations:
        return aid == gid
    elif aid in answers_frame:
        aclass = answers_frame[aid][0]
        gclass = golds_frame[gid][0]
        if do_soft_class:
            aclass = make_soft_classes(aclass)
            gclass = make_soft_classes(gclass)
        return aclass == gclass
    elif aid in answers_span:
        aclass = answers_span[aid][0]
        gclass = golds_span[gid][0]
        if do_soft_class:
            aclass = make_soft_classes(aclass)
            gclass = make_soft_classes(gclass)
        return aclass == gclass
    return False


def make_soft_classes(cur_class: str) -> str:
    cur_class = re.sub(r"^Positive\_r", "R", cur_class)
    cur_class = re.sub(r"^Negative\_r", "R", cur_class)
    cur_class = re.sub(r"^Transcription$", "Gene_expression", cur_class)
    return cur_class


def eq_args(
    aid: str,
    gid: str,
    a1_annotations: Dict[str, Tuple[str, int, int]],
    answers_span: Dict[str, Tuple[str, int, int]],
    answers_frame: Dict[str, Tuple[str, str, List[str]]],
    golds_span: Dict[str, Tuple[str, int, int]],
    golds_frame: Dict[str, Tuple[str, str, List[str]]],
    text: str,
    events_in_text: Dict[int, str],
    text_len: int,
    do_soft_class: bool,
    do_soft_args: bool,
    do_soft_span: bool,
    do_soft_overlap_span: bool,
) -> bool:
    ae_args = answers_frame[aid][2]
    ge_args = golds_frame[gid][2]

    if do_soft_args:
        while not re.search(r"^target\:", ae_args[-1]):
            ae_args.pop(-1)
        while not re.search(r"^target\:", ge_args[-1]):
            ge_args.pop(-1)

    if len(ge_args) != len(ae_args):
        return False

    # compare argument lists as ordered lists
    for i in range(0, len(ae_args)):
        aatype, aaid = ae_args[i].split(":")
        gatype, gaid = ge_args[i].split(":")

        if not do_soft_args:
            if aatype != gatype:
                return False
        # both have to be either t-entities or events
        if aaid[0] != gaid[0]:
            return False
        if re.search(r"^E", aaid) and not eq_revent(
            aaid,
            gaid,
            a1_annotations,
            answers_span,
            answers_frame,
            golds_span,
            golds_frame,
            text,
            events_in_text,
            text_len,
            do_soft_class,
            do_soft_args,
            do_soft_span,
            do_soft_overlap_span,
        ):
            return False
        if re.search(r"^T", aaid) and not eq_entity(
            aaid,
            gaid,
            a1_annotations,
            answers_span,
            answers_frame,
            golds_span,
            golds_frame,
            text,
            events_in_text,
            text_len,
            do_soft_class,
            do_soft_span,
            do_soft_overlap_span,
        ):
            return False
    return True


def eq_span(
    aid: str,
    gid: str,
    a1_annotations: Dict[str, Tuple[str, int, int]],
    answers_span: Dict[str, Tuple[str, int, int]],
    answers_frame: Dict[str, Tuple[str, str, List[str]]],
    golds_span: Dict[str, Tuple[str, int, int]],
    golds_frame: Dict[str, Tuple[str, str, List[str]]],
    text: str,
    events_in_text: Dict[int, str],
    text_len: int,
    do_soft_span: bool,
    do_soft_overlap: bool,
) -> bool:
    abeg = -1
    aend = -1
    gbeg = -2 if do_soft_span else -1
    gend = -2 if do_soft_span else -1

    if re.search(r"^T", aid) and aid in a1_annotations:
        return aid == gid

    if re.search(r"^T", aid):
        abeg = answers_span[aid][1]
        aend = answers_span[aid][2]
    elif re.search(r"^E", aid):
        abeg = answers_span[answers_frame[aid][1]][1]
        aend = answers_span[answers_frame[aid][1]][2]

    if re.search(r"^T", gid):
        gbeg = golds_span[gid][1]
        gend = golds_span[gid][2]
    elif re.search(r"^E", gid):
        gbeg = golds_span[golds_frame[gid][1]][1]
        gend = golds_span[golds_frame[gid][1]][2]

    if abeg < 0 or gbeg < 0:
        logging.error("failed to find the span: (" + aid + ", " + gid + ")")
        return False

    if do_soft_overlap:
        return (abeg <= gbeg and aend >= gbeg) or (gbeg <= abeg and gend >= abeg)
    elif do_soft_span:
        gbeg, gend = expand_span(gbeg, gend, text, events_in_text, text_len)
        return abeg >= gbeg and aend <= gend
    else:
        return abeg == gbeg and aend == gend


def expand_span(
    beg: int, end: int, text: str, events_in_text: Dict[int, str], text_len: int
) -> Tuple[int, int]:
    ebeg = beg - 2
    while (
        (ebeg >= 0)
        and (text[ebeg: ebeg + 1] not in [" ", ".", "!", "?", ",", "'", '"'])
        and (ebeg not in events_in_text or events_in_text[ebeg] != "E")
    ):
        ebeg -= 1
    ebeg += 1

    eend = end + 2
    while (
        (eend <= text_len)
        and (text[eend - 1: eend] not in [" ", ".", "!", "?", ",", "'", '"'])
        and (eend - 1 not in events_in_text or events_in_text[eend - 1] != "E")
    ):
        eend += 1
    eend -= 1
    return ebeg, eend


def eq_revent(
    aeid: str,
    geid: str,
    a1_annotations: Dict[str, Tuple[str, int, int]],
    answers_span: Dict[str, Tuple[str, int, int]],
    answers_frame: Dict[str, Tuple[str, str, List[str]]],
    golds_span: Dict[str, Tuple[str, int, int]],
    golds_frame: Dict[str, Tuple[str, str, List[str]]],
    text: str,
    events_in_text: Dict[int, str],
    text_len: int,
    do_soft_class: bool,
    do_soft_args: bool,
    do_soft_span: bool,
    do_soft_overlap_span: bool,
) -> bool:
    if not re.search(r"^E", aeid):
        logging.error("non-event annotation: " + aeid)
        return False
    if not re.search(r"^E", geid):
        logging.error("non-event annotation: " + geid)
        return False
    if (
        eq_class(
            aeid,
            geid,
            a1_annotations,
            answers_span,
            answers_frame,
            golds_span,
            golds_frame,
            do_soft_class,
        )
        and eq_span(
            aeid,
            geid,
            a1_annotations,
            answers_span,
            answers_frame,
            golds_span,
            golds_frame,
            text,
            events_in_text,
            text_len,
            do_soft_span,
            do_soft_overlap_span,
        )
        and eq_args(
            aeid,
            geid,
            a1_annotations,
            answers_span,
            answers_frame,
            golds_span,
            golds_frame,
            text,
            events_in_text,
            text_len,
            do_soft_class,
            do_soft_args,
            do_soft_span,
            do_soft_overlap_span,
        )
    ):
        return True
    else:
        return False


def eq_entity(
    aeid: str,
    geid: str,
    a1_annotations: Dict[str, Tuple[str, int, int]],
    answers_span: Dict[str, Tuple[str, int, int]],
    answers_frame: Dict[str, Tuple[str, str, List[str]]],
    golds_span: Dict[str, Tuple[str, int, int]],
    golds_frame: Dict[str, Tuple[str, str, List[str]]],
    text: str,
    events_in_text: Dict[int, str],
    text_len: int,
    do_soft_class: bool,
    do_soft_span: bool,
    do_soft_overlap_span: bool,
) -> bool:
    if not re.search(r"^T", aeid):
        logging.error("non-entity annotation: " + aeid)
        return False
    if not re.search(r"^T", geid):
        logging.error("non-entity annotation: " + geid)
        return False
    if eq_class(
        aeid,
        geid,
        a1_annotations,
        answers_span,
        answers_frame,
        golds_span,
        golds_frame,
        do_soft_class,
    ) and eq_span(
        aeid,
        geid,
        a1_annotations,
        answers_span,
        answers_frame,
        golds_span,
        golds_frame,
        text,
        events_in_text,
        text_len,
        do_soft_span,
        do_soft_overlap_span,
    ):
        return True
    else:
        return False


def count_match(
    answers_span: Dict[str, Tuple[str, int, int]],
    answers_frame: Dict[str, Tuple[str, str, List[str]]],
    golds_span: Dict[str, Tuple[str, int, int]],
    golds_frame: Dict[str, Tuple[str, str, List[str]]],
    a1_annotations: Dict[str, Tuple[str, int, int]],
    text: str,
    events_in_text: Dict[int, str],
    text_len: int,
    do_soft_class: bool,
    do_soft_args: bool,
    do_soft_span: bool,
    do_soft_overlap_span: bool,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    cnt_manswer = {}  # count matches of answer annotation instances
    cnt_matched_gold = {}  # count matches of gold annotation instances
    num_matched_answer = {}
    num_matched_gold = {}

    answer = []
    gold = []

    for key_a in answers_frame:
        answer.append(key_a)
        cnt_manswer[key_a] = 0

    for key_g in golds_frame:
        gold.append(key_g)
        cnt_matched_gold[key_g] = 0

    for aid in answer:  # for each answer
        for gid in gold:  # search for golds which match it
            if eq_event(
                aid,
                gid,
                a1_annotations,
                answers_span,
                answers_frame,
                golds_span,
                golds_frame,
                text,
                events_in_text,
                text_len,
                do_soft_class,
                do_soft_args,
                do_soft_span,
                do_soft_overlap_span,
            ):
                if aid not in cnt_manswer:
                    cnt_manswer[aid] = 0
                if gid not in cnt_matched_gold:
                    cnt_matched_gold[gid] = 0
                cnt_manswer[aid] += 1
                cnt_matched_gold[gid] += 1

    # update per-class statistics and store
    for a in answer:
        if cnt_manswer[a] > 0:
            a_class = answers_frame[a][0]
            if a_class not in num_matched_answer:
                num_matched_answer[a_class] = 0
            num_matched_answer[a_class] += 1

    for g in gold:
        if cnt_matched_gold[g] > 0:
            g_class = golds_frame[g][0]
            if g_class not in num_matched_gold:
                num_matched_gold[g_class] = 0
            num_matched_gold[g_class] += 1

    return num_matched_answer, num_matched_gold


def count_match_span(
    answers: Dict[str, Tuple[str, int, int]],
    golds: Dict[str, Tuple[str, int, int]],
    a1_annotations: Dict[str, Tuple[str, int, int]],
    text: str,
    events_in_text: Dict[int, str],
    text_len: int,
    do_soft_span: bool,
    do_soft_overlap_span: bool,
) -> Tuple[Dict[str, int], Dict[str, int]]:

    cnt_manswer = {}  # count matches of answer annotation instances
    cnt_matched_gold = {}  # count matches of gold annotation instances
    num_matched_answer_span = {}
    num_matched_gold_span = {}

    answer = []
    gold = []

    for key_a in answers:
        if re.search(r"^T", key_a):
            answer.append(key_a)
            cnt_manswer[key_a] = 0
    for key_g in golds:
        if re.search(r"^T", key_g):
            gold.append(key_g)
            cnt_matched_gold[key_g] = 0

    for aid in answer:  # for each answer
        for gid in gold:  # search for golds which match it
            if eq_span(
                aid,
                gid,
                a1_annotations,
                answers,
                {},
                golds,
                {},
                text,
                events_in_text,
                text_len,
                do_soft_span,
                do_soft_overlap_span,
            ):
                if aid not in cnt_manswer:
                    cnt_manswer[aid] = 0
                if gid not in cnt_matched_gold:
                    cnt_matched_gold[gid] = 0
                cnt_manswer[aid] += 1
                cnt_matched_gold[gid] += 1

    # update per-class statistics and store
    for a in answer:
        if cnt_manswer[a] > 0:
            a_class = answers[a][0]
            if a_class not in num_matched_answer_span:
                num_matched_answer_span[a_class] = 0
            num_matched_answer_span[a_class] += 1

    for g in gold:
        if cnt_matched_gold[g] > 0:
            g_class = golds[g][0]
            if g_class not in num_matched_gold_span:
                num_matched_gold_span[g_class] = 0
            num_matched_gold_span[g_class] += 1

    return num_matched_answer_span, num_matched_gold_span


# MAIN


@click.group()
def cli():
    pass


@cli.command()
@click.option("--hypo", required=True)
@click.option("--gold", required=True)
@click.option("--verbose", is_flag=True, default=False)
@click.option("--soft-span", is_flag=True, default=False)
@click.option("--soft-overlap-span", is_flag=True, default=False)
def deRE_evaluation(
    hypo: str, gold: str, verbose: bool, soft_span: bool, soft_overlap_span: bool
) -> None:
    files = glob.glob(hypo + "/*.ann")
    gold_dir = gold
    do_soft_class = False
    do_soft_args = False
    do_soft_span = soft_span
    do_soft_overlap_span = soft_overlap_span
    if verbose:
        logging.basicConfig(level=logging.INFO)

    target_class = [
        "positive",
        "negative",
        "neutral",
    ]

    tnum_gold = {}  # counts the number of gold events across files
    tnum_matched_gold = {}  # counts the number of matches of gold events across files
    tnum_answer = {}  # counts the number of hypothesis events across files
    tnum_matched_answer = {}  # counts the number of matches of hypothesis events across files
    tnum_gold_span = {}  # counts the number of gold spans across files
    tnum_matched_gold_span = {}  # counts the number of matches of gold spans across files
    tnum_answer_span = {}  # counts the number of hypothesis spans across files
    tnum_matched_answer_span = {}  # counts the number of matches of hypothesis spans across files

    for cl in target_class:  # initialization
        tnum_gold[cl] = 0
        tnum_matched_gold[cl] = 0
        tnum_answer[cl] = 0
        tnum_matched_answer[cl] = 0
        tnum_gold_span[cl] = 0
        tnum_matched_gold_span[cl] = 0
        tnum_answer_span[cl] = 0
        tnum_matched_answer_span[cl] = 0

    for f in files:
        f_dir, f_base = os.path.split(f)
        pmid = re.sub(r"(\S+)\.ann", "\\1", f_base)
        events_in_text = {}
        a1_annotations = {}
        gold = []
        equiv = {}

        # read text file
        text, text_len = read_text_file(gold_dir + "/" + pmid + ".txt")

        # read given annotations
        if os.path.exists(gold_dir + "/" + pmid + ".a1"):
            a1_annotations, events_in_text = read_a1_file(gold_dir + "/" + pmid + ".a1")

        # read gold annotations
        (
            gold_span_annotations,
            gold_frame_annotations,
            num_gold, num_gold_span,
            events_in_text,
            equiv
        ) = read_a2_file(gold_dir + "/" + pmid + ".a2.t1", events_in_text, equiv, "G")

        # read answers from system
        (
            answer_span_predictions,
            answer_frame_predictions,
            num_answer,
            num_answer_span,
            events_in_text,
            equiv
        ) = read_a2_file(f_dir + "/" + f_base, events_in_text, equiv, "A")

        # get number of matched spans
        num_matched_answer_span, num_matched_gold_span = count_match_span(
            answer_span_predictions,
            gold_span_annotations,
            a1_annotations,
            text,
            events_in_text,
            text_len,
            do_soft_span,
            do_soft_overlap_span,
        )

        # adjustment for duplication
        for cl in target_class:
            if cl not in num_matched_answer_span:
                num_matched_answer_span[cl] = 0
            if cl not in num_matched_gold_span:
                num_matched_gold_span[cl] = 0
            if num_matched_answer_span[cl] > num_matched_gold_span[cl]:
                num_danswer = num_matched_answer_span[cl] - num_matched_gold_span[cl]
                num_matched_answer_span[cl] -= num_danswer

        for cl in target_class:
            if cl in num_answer_span:
                tnum_answer_span[cl] += num_answer_span[cl]
                tnum_matched_answer_span[cl] += num_matched_answer_span[cl]
            if cl in num_gold_span:
                tnum_gold_span[cl] += num_gold_span[cl]
                tnum_matched_gold_span[cl] += num_matched_gold_span[cl]

        num_matched_answer, num_matched_gold = count_match(
            answer_span_predictions,
            answer_frame_predictions,
            gold_span_annotations,
            gold_frame_annotations,
            a1_annotations,
            text,
            events_in_text,
            text_len,
            do_soft_class,
            do_soft_args,
            do_soft_span,
            do_soft_overlap_span,
        )

        # adjustment for duplication
        for cl in target_class:
            if cl not in num_matched_answer:
                num_matched_answer[cl] = 0
            if cl not in num_matched_gold:
                num_matched_gold[cl] = 0
            if num_matched_answer[cl] > num_matched_gold[cl]:
                num_danswer = num_matched_answer[cl] - num_matched_gold[cl]
                num_answer[cl] -= num_danswer
                num_matched_answer[cl] -= num_danswer

        for cl in target_class:
            if cl in num_answer:
                tnum_answer[cl] += num_answer[cl]
                tnum_matched_answer[cl] += num_matched_answer[cl]
            if cl in num_gold:
                tnum_gold[cl] += num_gold[cl]
                tnum_matched_gold[cl] += num_matched_gold[cl]

    # CALCULATE AND OUTPUT RESULTS

    report_headline()

    print("-------------- SPAN EVALUATION ------------------")
    tnum_gold_span_value = 0
    tnum_matched_gold_span_value = 0
    tnum_answer_span_value = 0
    tnum_matched_answer_span_value = 0

    for cl in target_class:  # TODO: get target classes from specification!
        report(
            cl,
            tnum_gold_span[cl],
            tnum_matched_gold_span[cl],
            tnum_answer_span[cl],
            tnum_matched_answer_span[cl],
        )
        tnum_gold_span_value += tnum_gold_span[cl]
        tnum_matched_gold_span_value += tnum_matched_gold_span[cl]
        tnum_answer_span_value += tnum_answer_span[cl]
        tnum_matched_answer_span_value += tnum_matched_answer_span[cl]
    report(
        "=[TOTAL]=",
        tnum_gold_span_value,
        tnum_matched_gold_span_value,
        tnum_answer_span_value,
        tnum_matched_answer_span_value,
    )
    print("----------------------------------------------")
    print("-------------- EVENT EVALUATION ------------------")

    tnum_gold_value = 0
    tnum_matched_gold_value = 0
    tnum_answer_value = 0
    tnum_matched_answer_value = 0

    for cl in target_class:
        report(
            cl,
            tnum_gold[cl],
            tnum_matched_gold[cl],
            tnum_answer[cl],
            tnum_matched_answer[cl],
        )
        tnum_gold_value += tnum_gold[cl]
        tnum_matched_gold_value += tnum_matched_gold[cl]
        tnum_answer_value += tnum_answer[cl]
        tnum_matched_answer_value += tnum_matched_answer[cl]

    report(
        "=[EVENT-TOTAL]=",
        tnum_gold_value,
        tnum_matched_gold_value,
        tnum_answer_value,
        tnum_matched_answer_value,
    )
    print("----------------------------------------------")



if __name__ == "__main__":
    deRE_evaluation()
