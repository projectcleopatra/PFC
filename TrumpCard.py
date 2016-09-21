import requests
from requests.adapters import HTTPAdapter
import ssl

from requests.packages.urllib3 import PoolManager


class MyAdapter(HTTPAdapter):
    def init_poolmanager(self, connections, maxsize, block=False):
        self.poolmanager = PoolManager(num_pools=connections,
                                       maxsize=maxsize,
                                       block=block,
                                       ssl_version = ssl.PROTOCOL_TLSv1)



def jailbreakHTTPS(url: str):
    s = requests.Session()
    s.mount('https://', MyAdapter())
    return s.get(url)
