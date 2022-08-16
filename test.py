args = (
    "name",
    ("find", "path", "case"),
    ["arg", "ments"],
    {"key": "words", "key2": "words2"},
)


class A:
    def __init__(self) -> None:
        class Find:
            def __init__(self) -> None:
                class Path:
                    def case(self, *args, **kwargs):
                        return list(args) + list(kwargs.items())

                self.path = Path()

        self.find = Find()
        self.name = None


name = args[0]
callstack = args[1]
callargs = args[2]
callkwargs = args[3]

a_inst = A()
a_inst.name = name

cur = a_inst

# chain into the modules to find the appropriate function
for stackname in callstack:
    cur = getattr(cur, stackname)

print(cur(*callargs, **callkwargs))

"""
Every column needs to produce df of this format: 

Mv           Stock A     Stock B     Stock C
2020/1/31       1.3         2           4
2020/2/28...
2020/3/31...
2020/4/30...    NaN


Return         Stock A     Stock B     Stock C
2020/1/31       1.3         2           4
2020/2/28...
2020/3/31...
2020/4/30...    NaN

...


"""
