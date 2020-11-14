import sys
import sacrebleu
from statistics import mean
import bert_score
from bert_score import BERTScorer


def get_bert_score(hyp,ref,scorer):
    # hyp: hypothesis ref: reference scorer: Already created BERT Score object
    # Returns
    # F1: BERT-Score F1 between hypothesis and reference
    # Note: Some settings need to be done while creating the scorer object e.g whether to normalize by baseline or not, or which BERT model to use
    hyp = hyp.strip()
    ref = ref.strip()

    P, R, F1 = scorer.score([hyp,],[ref,])

    F1 = float(F1.data.cpu().numpy())

    return F1

def create_scorer():
    # Create scorer object for passing to get_bert_score
    scorer = BERTScorer(lang="en", rescale_with_baseline=True, model_type='roberta-base-cased')
    return scorer

def get_sent_bleu(hyp,ref):
    # hyp: hypothesis ref: reference
    # Returns:
    # bleu_score: Object with BLEU related information about sentence-level BLEU with exponential smoothing between hyp and ref
    hyp_line = hyp.strip()
    ref_line = ref.strip()

    bleu_score = sacrebleu.sentence_bleu(hyp_line,ref_line,smooth_method='exp')

    return bleu_score

def get_corpus_bleu(hyp,refs):
    hyp_line = hyp.strip()
    ref_lines = [ref.strip() for ref in refs]

    bleu_score = sacrebleu.corpus_bleu([hyp_line,],[[ref,] for ref in ref_lines],smooth_method='exp')

    return bleu_score

def get_corpus_bleu_parallel(hyp_list,refs_list):
    hyp_lines = [hyp.strip() for hyp in hyp_list]
    ref_lines_list = [[ref.strip() for ref in refs] for refs in refs_list]

    ref_size = len(ref_lines_list[0])

    ref_structure = [[] for i in range(ref_size)]
    for i in range(ref_size):
        ref_structure[i] = [ref_lines_list[j][i] for j in range(len(ref_lines_list))]

    bleu_score = sacrebleu.corpus_bleu(hyp_lines,ref_structure,smooth_method='exp')

    return bleu_score


def get_closest_sent_score(hyp_population,ref,metric="bleu_score",bert_scorer=None):
    # hyp_population: list of strings
    # ref: single string
    # metric: Choice of closeness metric (BERT score/BLEU score)
    # returns the highest metric (closest) between reference and any example from the hypothesis population

    highest_score = -2000.0
    highest_hyp = None
    highest_index = None

    mapping_obj = {}

    for i,hyp in enumerate(hyp_population):
        if metric=="bleu_score":
            score = get_sent_bleu(hyp,ref).score
        elif metric=="bert_score":
            score = get_bert_score(hyp,ref,bert_scorer)

        if score > highest_score:
            highest_index = i
            highest_hyp = hyp
            highest_score = score
            mapping_obj["index"] = highest_index
            mapping_obj["hyp"] = highest_hyp
            mapping_obj[metric] = highest_score

    return mapping_obj

def get_closest_sent_score_all(hyp_population,ref_population,metric="bleu_score",bert_scorer=None):

    closest_sent_score_all = {}

    mapping_objs = []

    for i,ref in enumerate(ref_population):
        mapping_obj = get_closest_sent_score(hyp_population,ref,metric=metric,bert_scorer=bert_scorer)
        mapping_objs.append(mapping_obj)

    #closest_sent_score_all["mapping_objs"] = mapping_objs
    # print(mapping_objs[0].keys())
    closest_sent_score_all["max"] = max([x[metric] for x in mapping_objs])
    #closest_sent_score_all["min"] = min([x[metric] for x in mapping_objs])
    closest_sent_score_all["mean"] = mean([x[metric] for x in mapping_objs])

    return closest_sent_score_all

def get_closest_sent_score_all_fast(hyp_population,ref_population,metric="bleu_score",bert_scorer=None):

    closest_sent_score_all = {}

    #mapping_objs = []

    #for i,ref in enumerate(ref_population):
    #    mapping_obj = get_closest_sent_score(hyp_population,ref,metric=metric,bert_scorer=bert_scorer)
    #    mapping_objs.append(mapping_obj)

    #closest_sent_score_all["mapping_objs"] = mapping_objs
    # print(mapping_objs[0].keys())
    #closest_sent_score_all["max"] = max([x[metric] for x in mapping_objs])
    #closest_sent_score_all["min"] = min([x[metric] for x in mapping_objs])
    #closest_sent_score_all["mean"] = mean([x[metric] for x in mapping_objs])

    total_bleu = 0.0
    for hyp_i in hyp_population:
        total_bleu += get_corpus_bleu(hyp_i,ref_population).score/len(ref_population)

    closest_sent_score_all["max"] = total_bleu/len(hyp_population)

    return closest_sent_score_all




def get_self_metric(hyp_population,metric="bleu_score",bert_scorer=None):
    # hyp_population: list of line strings, each string is a hypothesis/continuation
    # metric: Choice of closeness metric, bert_score or bleu_score
    # returns the self-metric [where metric is a closeness metric i.e bert_score or bleu_score]
    # Lower the self-metric, more the diversity

    closest_metrics = []

    for i,hyp_i in enumerate(hyp_population):
        closest_metric = -2000.0
        for j,hyp_j in enumerate(hyp_population):
            if j==i:
                continue

            if metric == "bleu_score":
                metric_score = get_sent_bleu(hyp_i,hyp_j).score
            elif metric == "bert_score":
                metric_score = get_bert_score(hyp_i,hyp_j,bert_scorer)

            if metric_score>closest_metric:
                closest_metric = metric_score
        closest_metrics.append(closest_metric)

    self_metric_score = mean(closest_metrics)

    return self_metric_score

def get_self_metric_corpus(hyp_population,metric="bleu_score",bert_scorer=None):
    # hyp_population: list of line strings, each string is a hypothesis/continuation
    # metric: Choice of closeness metric, bert_score or bleu_score
    # returns the self-metric [where metric is a closeness metric i.e bert_score or bleu_score]
    # Lower the self-metric, more the diversity
    closest_metrics = []

    for i,hyp_i in enumerate(hyp_population):
        rest_of_corpus = None

        if i == len(hyp_population)-1:
            rest_of_corpus = hyp_population[:-1]
        else:
            rest_of_corpus = hyp_population[:i] + hyp_population[i+1:]

        closest_metric = get_corpus_bleu(hyp_i,rest_of_corpus).score/len(rest_of_corpus)
        closest_metrics.append(closest_metric)

    self_metric_score = mean(closest_metrics)

    return self_metric_score

def get_self_metric_corpus_parallel(hyp_population,metric="bleu_score",bert_scorer=None):
    # hyp_population: list of line strings, each string is a hypothesis/continuation
    # metric: Choice of closeness metric, bert_score or bleu_score
    # returns the self-metric [where metric is a closeness metric i.e bert_score or bleu_score]
    # Lower the self-metric, more the diversity
    closest_metrics = []

    #hyps = []
    rest_of_corpuses = []

    hyps = [hyp_i for hyp_i in hyp_population]
    #rest_of_corpuses = [hyp_population[:-1] if i == (len(hyp_population)-1) else (hyp_population[:i] + hyp_population[i+1:]) for i in range(len(hyp_population))]
    
    #for i,hyp_i in enumerate(hyp_population):
    for i in range(len(hyp_population)):
        rest_of_corpus = None

        if i == len(hyp_population)-1:
            rest_of_corpus = hyp_population[:-1]
        else:
            rest_of_corpus = hyp_population[:i] + hyp_population[i+1:]
        
        #hyps.append(hyp_i)
        rest_of_corpuses.append(rest_of_corpus)
    
    self_metric_score = get_corpus_bleu_parallel(hyps,rest_of_corpuses).score/len(rest_of_corpuses[0])


    return self_metric_score



def get_unique_trigrams(hyp_population):
    # hyp_population: list of line strings, each string is a hypothesis/continuation
    # returns the unique trigram fraction in this population.
    # Higher the unique trigram fraction, more the diversity
    unique_trigrams = set()
    total_trigrams = 0

    for i,hyp_i in enumerate(hyp_population):
        hyp_i_words = hyp_i.strip().split()
        if len(hyp_i_words)>=3:
            total_trigrams += len(hyp_i_words)-2
            for j in range(len(hyp_i_words)-2):
                trigram = " ".join(hyp_i_words[j:j+2])
                unique_trigrams.add(trigram)

    unique_trigram_fraction = len(unique_trigrams)/(total_trigrams+1e-10)
    if total_trigrams == 0: unique_trigram_fraction = 0.0
    return unique_trigram_fraction

if __name__ == "__main__":

    hyp_a = "That year a great picnic was going to be the cause of their troubles."
    hyp_b = "She did not know that he would be so forward with his objections."
    hyp_c = "He looked forward to a great year ahead."
    ref_a = "Looking forward , a great year ahead eluded him."
    ref_b = "She seemed quite shocked at his forceful objections."
    ref_c = "A strange trip would lead to much troubles for him in the year that was to come."

    bleu_obj = get_sent_bleu(hyp_c,ref_a)
    print(bleu_obj.score)
    hyp_population = [hyp_a, hyp_b, hyp_c]
    mapping_obj = get_closest_sent_score(hyp_population,ref_a)
    print(mapping_obj)
    ref_population = [ref_a,ref_b,ref_c]

    closest_sent_bleu_all = get_closest_sent_score_all(hyp_population,ref_population)
    print(closest_sent_bleu_all)

    hyp_d = "A great picnic featuring year would precipitate a host of troubles for him in the year to come."
    hyp_population.append(hyp_d)
    self_bleu = get_self_metric(hyp_population)
    print(self_bleu)

    unique_trigram_fraction = get_unique_trigrams(hyp_population)
    print(unique_trigram_fraction)

    """
    scorer = create_scorer()
    hyp_e = "That was a great home run."
    ref_e = "Whoa seemed like a damn good shot."
    score = get_bert_score(hyp_e,ref_e,scorer)
    print(score)

    self_bert_score = get_self_metric(hyp_population,metric="bert_score",bert_scorer=scorer)
    print(self_bert_score)

    closest_sent_bert_score_all = get_closest_sent_score_all(hyp_population,ref_population,metric="bert_score",bert_scorer=scorer)
    print(closest_sent_bert_score_all)
    """

