from multiprocessing import Process

class CustomProcess(Process):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
    
    def run(self):
        print(*self.args)
        return 1

args = {'a': 1, 'b': 2}

p = CustomProcess(args=args)
res = p.run()
print(res)