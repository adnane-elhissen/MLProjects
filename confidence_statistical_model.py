#! /usr/bin/env python3

from tools import probabilities as pb
import random


def bootstrap_sample(data):
    return [random.choice(data) for _ in data]

def bootstrap_statistics(data,stats_fn,num_samples):
    return [stats_fn(bootstrap_sample(data)) for _ in range(num_samples)]

"""
random.seed(0)
bootstrap_betas = bootstrap_statistics (data, estimate_sample_beta,100)

bootstrap_standard_errors = [ct.standard_deviation([beta[i] for beta in bootstrap_betas]) for i in range(4)]
"""
def p_value(beta_h_i,sigma_h_i):
    return pb.normal_cdf(beta_h_i/sigma_h_i)