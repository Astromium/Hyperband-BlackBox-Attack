from typing import List

def get_feature_family(features_l: List[str], family: str) -> List[str]:
        return list(filter(lambda el: el.startswith(family), features_l))


def get_constraints_features(X):
    features = X.columns.to_list()
    feature_families = ['icmp_sum_s_', 'udp_sum_s_', 'tcp_sum_s_', 'bytes_in_sum_s_', 'bytes_out_sum_s_', 'icmp_sum_d_', 'udp_sum_d_', 'tcp_sum_d_', 'bytes_in_sum_d_', 'bytes_out_sum_d_', 'pkts_out_sum_s_', 'pkts_out_sum_d_']
    families = [get_feature_family(features, f) for f in feature_families]
    indices = []
    for family in families:
        for f in family:
            indices.append(X.columns.get_loc(f))

    return indices