class Bank:
    def __init__(self):
        # account: balances, activity
        self.accounts = dict()

    def process_query(self, q):

        query2func = {
            "CREATE_ACCOUNT": self.create_account,
            "DEPOSIT": self.deposit,
            "PAY": self.pay,
            "TOP_ACTIVITY": self.top_activity,
        }

        return query2func[q[0]](*q[1:])

    def account_exists(self, name: str):
        if name in self.accounts:
            return True
        else:
            return False

    def create_account(self, id: str, name: str, **kwargs):
        if self.account_exists(name):
            return False
        else:
            self.accounts[name] = list([0, 0])
            return True

    def deposit(self, id: str, name: str, amt: str, **kwargs):
        if not self.account_exists(name):
            return ""
        else:
            self.accounts[name][0] += int(amt)
            self.accounts[name][1] += int(amt)
            return self.accounts[name][0]

    def pay(self, id: str, name: str, amt: str, **kwargs):
        if not self.account_exists(name):
            return ""
        else:
            if self.accounts[name][0] >= int(amt):
                self.accounts[name][0] -= int(amt)
                self.accounts[name][1] += int(amt)
                return self.accounts[name][0]
            else:
                return ""

    def top_activity(self, id: str, n: str):
        # import pdb; pdb.set_trace()

        acct_act = [-v[1] for v in self.accounts.values()]
        acct_names = list(self.accounts.keys())

        out = sorted(
            zip(acct_act, acct_names), key=lambda x: ((x[0]), x[1])
        )

        # out = sorted(
        #     zip(list(self.accounts.items()), list(self.accounts.keys())),
        #     key=lambda x: (-(x[0][1]), x[1])
        # )

        out = [[o[1], -o[0]] for o in out[:int(n)]]
        return ", ".join(map(self.top_activity_format, out))

    def top_activity_format(self, tpl):
        name, attr = tpl
        return "{}({})".format(name, attr)


bank = Bank()


def solution(queries):
    outputs = list()
    for q in queries:

        out = bank.process_query(q)
        if isinstance(out, bool):
            outputs.append(str(out).lower())
        else:
            outputs.append(str(out))

    return outputs


if __name__ == '__main__':

    out = solution(
        queries=
        [["CREATE_ACCOUNT", "1", "account1"],
         ["CREATE_ACCOUNT", "2", "account2"],
         ["CREATE_ACCOUNT", "3", "account3"],
         ["DEPOSIT", "4", "account1", "1000"],
         ["DEPOSIT", "5", "account2", "1000"],
         ["DEPOSIT", "6", "account3", "1000"],
         ["PAY", "7", "account2", "100"],
         ["PAY", "8", "account2", "100"],
         ["PAY", "9", "account3", "100"],
         ["TOP_ACTIVITY", "10", "3"]]
    )
    print(out)