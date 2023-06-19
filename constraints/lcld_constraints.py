from .relation_constraint import Constant as Co
from .relation_constraint import Feature as Fe
from .relation_constraint import SafeDivision

def get_relation_constraints():

        int_rate = Fe("int_rate") / Co(1200)
        term = Fe("term")
        installment = Fe("loan_amnt") * (
            (int_rate * ((Co(1) + int_rate) ** term))
            / ((Co(1) + int_rate) ** term - Co(1))
        )
        g1 = Fe("installment") == installment

        g2 = Fe("open_acc") <= Fe("total_acc")

        g3 = Fe("pub_rec_bankruptcies") <= Fe("pub_rec")

        g4 = (Fe("term") == Co(36)) | (Fe("term") == Co(60))

        g5 = Fe("ratio_loan_amnt_annual_inc") == (
            Fe("loan_amnt") / Fe("annual_inc")
        )

        g6 = Fe("ratio_open_acc_total_acc") == (
            Fe("open_acc") / Fe("total_acc")
        )

        # g7 was diff_issue_d_earliest_cr_line
        # g7 is not necessary in v2
        # issue_d and d_earliest cr_line are replaced
        # by month_since_earliest_cr_line

        g8 = Fe("ratio_pub_rec_month_since_earliest_cr_line") == (
            Fe("pub_rec") / Fe("month_since_earliest_cr_line")
        )

        g9 = Fe("ratio_pub_rec_bankruptcies_month_since_earliest_cr_line") == (
            Fe("pub_rec_bankruptcies") / Fe("month_since_earliest_cr_line")
        )

        g10 = Fe("ratio_pub_rec_bankruptcies_pub_rec") == SafeDivision(
            Fe("pub_rec_bankruptcies"), Fe("pub_rec"), Co(-1)
        )

        return [g1, g2, g3, g4, g5, g6, g8, g9, g10]