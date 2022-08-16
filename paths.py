import getpass as _getpass
import platform as _platform

_paths = {
    "taeheecho@myubuntu:Linux": "/usr/local/etf",
    "jaehoon@jaehoon-X299-WU8:Linux": "/home/jaehoon/QRAFT/data",
    "jaehoon@jaehoon-B450M-AORUS-ELITE:Linux": "/home/sronly/sr-storage",
    "sronly@ubuntu:Linux": "/home/sronly/sr-storage",
    "sronly@DGX-A100:Linux": "/home/sronly/sr-storage",
    "sronly@node-dgx:Linux" : "/home/sronly/sr-storage",
    "jaehoon@Jaehoonui-MacBookPro.local:Darwin": "/Users/jaehoon/QRAFT/data",
    "youngmin@DGX-1V:Linux": "/raid/youngmin/sr-storage",
    'marketing@DESKTOP-91GJTDB:Windows': r"C:\Users\marketing\Documents"
}

_identifier = f"{_getpass.getuser()}@{_platform.node()}:{_platform.system()}"
DATA_DIR = _paths[_identifier]

NEPTUNE_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiN2VlMDM5ZGItMWVjZi00N2Q2LTk3N2EtMDlhN2VjMGI1YjdjIn0="


if __name__ == "__main__":
    print(_identifier)
