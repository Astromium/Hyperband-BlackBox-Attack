from typing import List
from .relation_constraint import Constant as Co
from .relation_constraint import Feature as Fe
from .relation_constraint import MathOperation, SafeDivision


def get_relation_constraints(x):

    features = x.columns.to_list()

    def get_feature_family(features_l: List[str], family: str) -> List[str]:
        return list(filter(lambda el: el.startswith(family), features_l))

    def sum_list_feature(features_l: List[str]) -> MathOperation:
        out = Fe(features_l[0])
        for el in features_l[1:]:
            out = out + Fe(el)
        return out

    def sum_feature_family(features_l: List[str], family: str):
        return sum_list_feature(get_feature_family(features_l, family))

    g1 = (
        sum_feature_family(features, "icmp_sum_s_")
        + sum_feature_family(features, "udp_sum_s_")
        + sum_feature_family(features, "tcp_sum_s_")
    ) == (
        sum_feature_family(features, "bytes_in_sum_s_")
        + sum_feature_family(features, "bytes_out_sum_s_")
    )

    g2 = (
        sum_feature_family(features, "icmp_sum_d_")
        + sum_feature_family(features, "udp_sum_d_")
        + sum_feature_family(features, "tcp_sum_d_")
    ) == (
        sum_feature_family(features, "bytes_in_sum_d_")
        + sum_feature_family(features, "bytes_out_sum_d_")
    )

    g_packet_size = []
    for e in ["s", "d"]:
        # -1 cause ignore last OTHER features
        bytes_outs = get_feature_family(features, f"bytes_out_sum_{e}_")[
            :-1
        ]
        pkts_outs = get_feature_family(features, f"pkts_out_sum_{e}_")[:-1]
        if len(bytes_outs) != len(pkts_outs):
            raise Exception("len(bytes_out) != len(pkts_out)")

        # Tuple of list to list of tuples
        for byte_out, pkts_out in list(zip(bytes_outs, pkts_outs)):
            g = SafeDivision(Fe(byte_out), Fe(pkts_out), Co(0.0)) <= Co(
                1500
            )
            g_packet_size.append(g)

    g_min_max_sum = []
    for e_1 in ["bytes_out", "pkts_out", "duration"]:
        for port in [
            "1",
            "3",
            "8",
            "10",
            "21",
            "22",
            "25",
            "53",
            "80",
            "110",
            "123",
            "135",
            "138",
            "161",
            "443",
            "445",
            "993",
            "OTHER",
        ]:
            for e_2 in ["d", "s"]:
                g_min_max_sum.extend(
                    [
                        Fe(f"{e_1}_max_{e_2}_{port}")
                        <= Fe(f"{e_1}_sum_{e_2}_{port}"),
                        Fe(f"{e_1}_min_{e_2}_{port}")
                        <= Fe(f"{e_1}_sum_{e_2}_{port}"),
                        Fe(f"{e_1}_min_{e_2}_{port}")
                        <= Fe(f"{e_1}_max_{e_2}_{port}"),
                    ]
                )

    return [g1, g2] + g_packet_size + g_min_max_sum
